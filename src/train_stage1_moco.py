"""
Stage 1: MoCo v3 Contrastive Pretraining
Unsupervised feature learning on low-light images
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
from tqdm import tqdm
import time
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from src.data_loader import LowLightDataset
from model.encoder_moco import MoCoV3Encoder
from utils import *

import torch.nn.functional as F

try:
    from torch.amp import GradScaler  # type: ignore[attr-defined]
    from torch.amp.autocast_mode import autocast
    _GRADSCALER_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore
    from torch.cuda.amp import autocast
    _GRADSCALER_NEW_API = False


def gamma_exposure_jitter(
    images: torch.Tensor,
    gamma_range=(0.6, 1.5),
    exposure_range=(-0.15, 0.15),
):
    """Apply per-sample gamma and exposure jitter."""
    if images is None:
        return images
    if gamma_range is None and exposure_range is None:
        return images
    out = images
    if gamma_range is not None:
        gamma = torch.empty(images.size(0), 1, 1, 1, device=images.device).uniform_(
            gamma_range[0], gamma_range[1]
        )
        out = torch.clamp(out, 1e-6, 1.0)
        out = out.pow(1.0 / gamma)
    if exposure_range is not None:
        exposure = torch.empty(images.size(0), 1, 1, 1, device=images.device).uniform_(
            exposure_range[0], exposure_range[1]
        )
        out = torch.clamp(out + exposure, 0.0, 1.0)
    return out


def build_linear_probe_dataset(dataset_cfg: dict, batch_size: int, num_workers: int):
    """Create a dataloader for periodic linear probe evaluation."""
    probe_path = dataset_cfg.get("dataset")
    if not probe_path or not os.path.exists(probe_path):
        raise FileNotFoundError(f"Linear probe dataset not found: {probe_path}")
    from torchvision import transforms, datasets

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(probe_path, transform=tfm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.classes


def run_linear_probe_eval(
    model: MoCoV3Encoder,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> float:
    """Freeze encoder and train a logistic regression probe on-the-fly."""
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            emb, _ = model.encoder_q(images, return_intermediate=False)
            emb = torch.nn.functional.normalize(emb, dim=1)
            feats.append(emb.cpu().numpy())
            labels.append(target.numpy())
            if max_samples and sum(len(x) for x in labels) >= max_samples:
                break
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    if max_samples and len(labels) > max_samples:
        idx = np.random.choice(len(labels), size=max_samples, replace=False)
        feats = feats[idx]
        labels = labels[idx]
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=2000, multi_class="auto"))
    clf.fit(feats, labels)
    preds = clf.predict(feats)
    acc = float(accuracy_score(labels, preds))
    model.train()
    return acc




def bidirectional_contrastive_loss(q: torch.Tensor, k: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(q.size(0), device=q.device)
    logits_qk = torch.matmul(q, k.t()) / temperature
    logits_kq = torch.matmul(k, q.t()) / temperature
    loss_qk = F.cross_entropy(logits_qk, labels)
    loss_kq = F.cross_entropy(logits_kq, labels)
    return 0.5 * (loss_qk + loss_kq)

def train_moco(config):
    """
    Train MoCo v3 encoder on low-light images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    seed = config['training'].get('seed')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def compute_grad_norm(parameters):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5 if total_norm > 0 else 0.0
    
    # Create experiment directory
    exp_dirs = create_exp_dir(config['output_dir'], config['exp_name'])
    
    # Save config
    save_config(config, os.path.join(exp_dirs['logs'], 'config.yaml'))
    
    # Initialize TensorBoard
    writer = SummaryWriter(exp_dirs['logs'])

    monitor_cfg = config.get('monitoring', {})
    metrics_interval = monitor_cfg.get('metrics_interval', 20)
    log_grad_norm = monitor_cfg.get('log_grad_norm', True)
    log_queue_stats = monitor_cfg.get('log_queue_stats', True)
    log_memory = monitor_cfg.get('log_memory', False)

    metrics_csv_path = os.path.join(exp_dirs['logs'], 'metrics.csv')
    with open(metrics_csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            'epoch', 'step', 'loss',
            'pos_sim_mean', 'pos_sim_std',
            'neg_sim_mean', 'neg_sim_std',
            'grad_norm', 'lr',
            'iter_time_ms', 'compute_time_ms', 'data_time_ms',
            'speed_imgs_sec',
            'mid_loss', 'illum_loss', 'temperature'
        ])
    
    # Create dataset and dataloader
    print("Loading datasets...")
    train_dataset = LowLightDataset(
        root_dir=config['dataset']['root_dir'],
        paired=(config['dataset'].get('mode', 'unpaired') == 'paired'),
        split='train',
        img_size=config['dataset']['img_size'],
        augment=True,
        normalize=True,
        return_pair=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    aug_cfg = config.get('augmentation', {})
    gamma_range = tuple(aug_cfg.get('gamma_range', (0.6, 1.5))) if aug_cfg.get('gamma_range', None) else None
    exposure_range = tuple(aug_cfg.get('exposure_range', (-0.15, 0.15))) if aug_cfg.get('exposure_range', None) else (-0.15, 0.15)
    apply_gamma = aug_cfg.get('enable_gamma', True)

    mid_cfg = config.get('mid_level_contrast', {})
    mid_enabled = bool(mid_cfg.get('enabled', False) and mid_cfg.get('layers'))
    mid_layers = mid_cfg.get('layers', []) if mid_enabled else []
    mid_weight = mid_cfg.get('weight', 0.1) if mid_enabled else 0.0
    mid_temperature = mid_cfg.get('temperature', 0.1)

    illum_cfg = config.get('illumination', {})
    illumination_iter = None
    illumination_weight = illum_cfg.get('weight', 0.2)
    illumination_frequency = max(1, int(illum_cfg.get('frequency', 1)))
    if illum_cfg.get('enabled', False):
        illum_root = illum_cfg.get('root_dir')
        if illum_root is None:
            raise ValueError("illumination.enabled is True but root_dir is not specified")
        illum_datasets = illum_cfg.get('datasets') or [""]
        illum_img_size = illum_cfg.get('img_size', config['dataset']['img_size'])
        illum_list = []
        for name in illum_datasets:
            root_path = os.path.join(illum_root, name) if name else illum_root
            if not os.path.exists(root_path):
                print(f"[Warning] Illumination dataset not found: {root_path}")
                continue
            ds = LowLightDataset(
                root_dir=root_path,
                paired=True,
                split='train',
                img_size=illum_img_size,
                augment=False,
                normalize=True,
                return_pair=False
            )
            if len(ds) > 0:
                illum_list.append(ds)
        if len(illum_list) == 0:
            print("[Warning] No valid illumination datasets found; disabling illumination-aware loss.")
            illumination_weight = 0.0
        else:
            if len(illum_list) == 1:
                illum_dataset = illum_list[0]
            else:
                from torch.utils.data import ConcatDataset
                illum_dataset = ConcatDataset(illum_list)
            illum_loader = DataLoader(
                illum_dataset,
                batch_size=illum_cfg.get('batch_size', config['training']['batch_size']),
                shuffle=True,
                num_workers=config['training']['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            illumination_iter = cycle(illum_loader)
            freq_desc = illumination_frequency if illumination_frequency else 1
            print(f"Illumination pairs enabled (weight={illumination_weight}, freq={freq_desc}).")
    else:
        illumination_weight = 0.0

    temp_cfg = config.get('dynamic_temperature', {})
    dynamic_temp_enabled = temp_cfg.get('enabled', False)
    tau_base = config['model'].get('temperature', 0.07)
    tau_min = temp_cfg.get('min', 0.04)
    tau_max = temp_cfg.get('max', 0.2)
    tau_momentum = temp_cfg.get('momentum', 0.9)
    running_tau = tau_base

    linear_probe_loader = None
    linear_probe_freq = 0
    linear_probe_cfg = monitor_cfg.get('linear_probe', {})
    if linear_probe_cfg.get('enabled', False):
        try:
            linear_probe_loader, probe_classes = build_linear_probe_dataset(
                linear_probe_cfg,
                linear_probe_cfg.get('batch_size', 64),
                linear_probe_cfg.get('num_workers', 0)
            )
            linear_probe_freq = max(1, linear_probe_cfg.get('frequency', 20))
            writer.add_text('linear_probe/classes', "\n".join(probe_classes))
            print(f"Linear probe evaluation enabled (freq={linear_probe_freq} epochs).")
        except Exception as exc:
            print(f"[Warning] Failed to initialise linear probe dataset: {exc}")
            linear_probe_loader = None
            linear_probe_freq = 0
    
    # Initialize model
    print("Initializing MoCo v3 model...")
    model = MoCoV3Encoder(
        base_encoder=config['model']['base_encoder'],
        feat_dim=config['model']['feat_dim'],
        queue_size=config['model']['queue_size'],
        momentum=config['model']['momentum'],
        temperature=config['model']['temperature'],
        mid_layers=mid_layers if mid_enabled else None
    ).to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr'],
        momentum=0.9,
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    warmup_lr = config['training'].get('warmup_lr', config['training']['lr'] * 0.1)
    total_epochs = config['training']['epochs']
    if warmup_epochs > 0 and warmup_epochs < total_epochs:
        start_factor = max(1e-6, warmup_lr / config['training']['lr'])
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Training loop
    print("Starting training...")
    global_step = 0
    
    max_steps_per_epoch = int(config['training'].get('max_steps_per_epoch', 0) or 0)

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = AverageMeter()
        epoch_pos = AverageMeter()
        epoch_neg = AverageMeter()
        epoch_speed = AverageMeter()
        data_meter = AverageMeter()
        grad_meter = AverageMeter()
        illum_meter = AverageMeter()
        mid_meter = AverageMeter()
        num_batches = len(train_loader)
        if max_steps_per_epoch > 0:
            num_batches = min(num_batches, max_steps_per_epoch)
        train_iterator = iter(train_loader)
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        steps_this_epoch = 0
        for batch_idx in pbar:
            data_wait_start = time.time()
            try:
                batch = next(train_iterator)
            except StopIteration:
                break
            data_wait = time.time() - data_wait_start
            iter_start = time.time()
            im_q = batch['low'].to(device)
            im_k = batch['low_pair'].to(device) if 'low_pair' in batch else batch['low'].to(device)

            if apply_gamma and gamma_range is not None:
                im_q = gamma_exposure_jitter(im_q, gamma_range, exposure_range)
                im_k = gamma_exposure_jitter(im_k, gamma_range, exposure_range)

            illum_loss_val = torch.tensor(0.0, device=device)
            illum_low = illum_high = None
            mid_loss_val = torch.tensor(0.0, device=device)
            use_illumination = (
                illumination_iter is not None
                and illumination_weight > 0.0
                and (global_step % illumination_frequency == 0)
            )
            if use_illumination:
                illum_batch = next(illumination_iter)
                illum_low = illum_batch['low'].to(device)
                illum_high = illum_batch['high'].to(device)
                if apply_gamma and gamma_range is not None:
                    illum_low = gamma_exposure_jitter(illum_low, gamma_range, exposure_range)
                    illum_high = gamma_exposure_jitter(illum_high, gamma_range, exposure_range)
            else:
                illum_batch = None

            amp_enabled = scaler.is_enabled()
            autocast_ctx = (
                autocast('cuda', enabled=amp_enabled)
                if _GRADSCALER_NEW_API
                else autocast(device_type='cuda', enabled=amp_enabled)
            )
            with autocast_ctx:
                logits, labels, extras = model(im_q, im_k)
                loss = criterion(logits, labels)

                if mid_enabled:
                    q_feats = extras.get("mid_q", {}) if extras else {}
                    k_feats = extras.get("mid_k", {}) if extras else {}
                    mid_losses = []
                    for layer in mid_layers:
                        if layer in q_feats and layer in k_feats:
                            mid_losses.append(
                                bidirectional_contrastive_loss(
                                    q_feats[layer], k_feats[layer], mid_temperature
                                )
                            )
                    if mid_losses:
                        if len(mid_losses) == 1:
                            mid_loss_val = mid_losses[0]
                        else:
                            mid_loss_val = torch.stack(mid_losses).mean()

            illumination_active = (
                illum_batch is not None and illum_low is not None and illum_high is not None
            )
            total_loss = loss
            if illumination_active:
                enc_low, _ = model.encoder_q(illum_low, return_intermediate=False)
                enc_high, _ = model.encoder_q(illum_high, return_intermediate=False)
                emb_low = F.normalize(enc_low, dim=1)
                emb_high = F.normalize(enc_high, dim=1)
                illum_loss_val = (1.0 - (emb_low * emb_high).sum(dim=1)).mean()

            if illumination_active and illumination_weight > 0.0:
                total_loss = total_loss + illumination_weight * illum_loss_val
            if mid_enabled and mid_loss_val.item() > 0 and mid_weight > 0.0:
                total_loss = total_loss + mid_weight * mid_loss_val

            optimizer.zero_grad(set_to_none=True)
            current_grad_norm = 0.0
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                if log_grad_norm:
                    scaler.unscale_(optimizer)
                    current_grad_norm = compute_grad_norm(model.encoder_q.parameters())
                    grad_meter.update(current_grad_norm, 1)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if log_grad_norm:
                    current_grad_norm = compute_grad_norm(model.encoder_q.parameters())
                    grad_meter.update(current_grad_norm, 1)
                optimizer.step()

            epoch_loss.update(total_loss.item(), im_q.size(0))
            if illumination_active:
                illum_meter.update(illum_loss_val.item(), illum_low.size(0))
            if mid_enabled:
                mid_meter.update(mid_loss_val.item(), im_q.size(0))

            iter_compute_time = time.time() - iter_start
            total_iter_time = data_wait + iter_compute_time
            iter_time_ms = total_iter_time * 1000.0
            compute_time_ms = iter_compute_time * 1000.0
            data_time_ms = max(data_wait, 0.0) * 1000.0
            imgs_per_sec = im_q.size(0) / total_iter_time if total_iter_time > 0 else 0.0
            data_meter.update(data_time_ms, 1)
            epoch_speed.update(imgs_per_sec, im_q.size(0))

            if dynamic_temp_enabled and model.last_pos_sim is not None and model.last_neg_logits is not None:
                pos_std = model.last_pos_sim.std()
                neg_std = model.last_neg_logits.std()
                if torch.isfinite(pos_std) and torch.isfinite(neg_std) and neg_std > 0:
                    target_tau = float(torch.clamp(tau_base * (pos_std / (neg_std + 1e-6)), tau_min, tau_max))
                    running_tau = tau_momentum * running_tau + (1 - tau_momentum) * target_tau
                    model.set_temperature(running_tau)

            if metrics_interval and (global_step % metrics_interval == 0):
                pos_mean = pos_std = neg_mean = neg_std = 0.0
                if model.last_pos_sim is not None:
                    pos_mean = model.last_pos_sim.mean().item()
                    pos_std = model.last_pos_sim.std().item()
                    writer.add_scalar('train/pos_sim', pos_mean, global_step)
                    writer.add_scalar('train/pos_sim_std', pos_std, global_step)
                    epoch_pos.update(pos_mean, im_q.size(0))
                if model.last_neg_logits is not None:
                    neg_mean = model.last_neg_logits.mean().item()
                    neg_std = model.last_neg_logits.std().item()
                    writer.add_scalar('train/neg_sim', neg_mean, global_step)
                    writer.add_scalar('train/neg_sim_std', neg_std, global_step)
                    epoch_neg.update(neg_mean, im_q.size(0))

                if log_queue_stats:
                    col_norm = model.queue.norm(dim=0)
                    writer.add_scalar('train/queue_norm', col_norm.mean().item(), global_step)
                    writer.add_scalar('train/queue_std', col_norm.std().item(), global_step)

                if log_grad_norm:
                    writer.add_scalar('train/grad_norm', current_grad_norm, global_step)

                writer.add_scalar('train/iter_loss', total_loss.item(), global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('train/iter_time_ms', iter_time_ms, global_step)
                writer.add_scalar('train/compute_time_ms', compute_time_ms, global_step)
                writer.add_scalar('train/data_time_ms', data_time_ms, global_step)
                writer.add_scalar('train/throughput', imgs_per_sec, global_step)
                if illumination_active and illumination_weight > 0.0:
                    writer.add_scalar('train/illum_loss', illum_loss_val.item(), global_step)
                if mid_enabled:
                    writer.add_scalar('train/mid_loss', mid_loss_val.item(), global_step)
                if dynamic_temp_enabled:
                    writer.add_scalar('train/temperature', model.temperature, global_step)
                with open(metrics_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        epoch + 1,
                        global_step,
                        total_loss.item(),
                        pos_mean,
                        pos_std,
                        neg_mean,
                        neg_std,
                        current_grad_norm if log_grad_norm else '',
                        optimizer.param_groups[0]['lr'],
                        iter_time_ms,
                        compute_time_ms,
                        data_time_ms,
                        imgs_per_sec,
                        mid_loss_val.item() if mid_enabled else 0.0,
                        illum_loss_val.item() if illumination_active else 0.0,
                        model.temperature if dynamic_temp_enabled else tau_base
                    ])

            pbar.set_postfix({'loss': f'{epoch_loss.avg:.4f}'})
            global_step += 1
            steps_this_epoch += 1
            if max_steps_per_epoch > 0 and steps_this_epoch >= max_steps_per_epoch:
                break
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch metrics
        avg_pos = epoch_pos.avg if epoch_pos.count > 0 else 0.0
        avg_neg = epoch_neg.avg if epoch_neg.count > 0 else 0.0
        avg_speed = epoch_speed.avg if epoch_speed.count > 0 else 0.0
        avg_data_wait = data_meter.avg if data_meter.count > 0 else 0.0
        avg_grad = grad_meter.avg if grad_meter.count > 0 else 0.0
        avg_illum = illum_meter.avg if illum_meter.count > 0 else 0.0
        avg_mid = mid_meter.avg if mid_meter.count > 0 else 0.0

        print(f"Epoch {epoch+1} - Average Loss: {epoch_loss.avg:.4f}")
        if epoch_pos.count > 0:
            print(f"  Avg positive similarity: {avg_pos:.4f}")
        if epoch_neg.count > 0:
            print(f"  Avg negative similarity: {avg_neg:.4f}")
        if epoch_speed.count > 0:
            print(f"  Avg throughput: {avg_speed:.2f} img/s")
        if data_meter.count > 0:
            print(f"  Avg data wait: {avg_data_wait:.2f} ms")
        if log_grad_norm and grad_meter.count > 0:
            print(f"  Avg gradient norm: {avg_grad:.4f}")
        if illum_meter.count > 0:
            print(f"  Avg illumination loss: {avg_illum:.4f}")
        if mid_meter.count > 0:
            print(f"  Avg mid-level contrast loss: {avg_mid:.4f}")

        writer.add_scalar('Loss/epoch', epoch_loss.avg, epoch)
        writer.add_scalar('train/epoch_pos_sim', avg_pos, epoch)
        writer.add_scalar('train/epoch_neg_sim', avg_neg, epoch)
        writer.add_scalar('train/epoch_speed', avg_speed, epoch)
        if data_meter.count > 0:
            writer.add_scalar('train/epoch_data_wait', avg_data_wait, epoch)
        if log_grad_norm and grad_meter.count > 0:
            writer.add_scalar('train/epoch_grad_norm', avg_grad, epoch)
        if illum_meter.count > 0:
            writer.add_scalar('train/epoch_illum_loss', avg_illum, epoch)
        if mid_meter.count > 0:
            writer.add_scalar('train/epoch_mid_loss', avg_mid, epoch)
        if dynamic_temp_enabled:
            writer.add_scalar('train/epoch_temperature', model.temperature, epoch)

        log_dir = exp_dirs['logs']
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "train_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch+1} - Loss: {epoch_loss.avg:.4f}, "
                f"PosSim: {avg_pos:.4f}, NegSim: {avg_neg:.4f}, "
                f"Speed: {avg_speed:.2f} img/s, DataWait: {avg_data_wait:.2f} ms, "
                f"GradNorm: {avg_grad:.4f}, "
                f"IllumLoss: {avg_illum:.4f}, MidLoss: {avg_mid:.4f}\n"
            )

        if linear_probe_loader is not None and linear_probe_freq > 0 and (epoch + 1) % linear_probe_freq == 0:
            probe_acc = run_linear_probe_eval(
                model,
                linear_probe_loader,
                device,
                max_samples=linear_probe_cfg.get('max_samples', 2000)
            )
            writer.add_scalar('eval/linear_probe_top1', probe_acc, epoch + 1)
            print(f"  Linear probe top-1 accuracy: {probe_acc:.4f}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"  Linear probe top-1: {probe_acc:.4f}\n")

        # Save checkpoint
        if (epoch + 1) % config['training']['save_freq'] == 0:
            checkpoint_path = os.path.join(
                exp_dirs['checkpoints'],
                f'moco_pretrain_epoch_{epoch+1}.pth'
            )
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss.avg,
            }, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(exp_dirs['checkpoints'], 'moco_pretrain_final.pth')
    save_checkpoint({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo v3 Pretraining')
    parser.add_argument('--config', type=str, default='../configs/train_stage1.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train_moco(config)
