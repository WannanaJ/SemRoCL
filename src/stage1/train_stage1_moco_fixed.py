"""
Stage 1: MoCo v3 Contrastive Pretraining
Unsupervised feature learning on low-light images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm

from src.data_loader import LowLightDataset
from model.encoder_moco import MoCoV3Encoder
from utils import *


def train_moco(config):
    """
    Train MoCo v3 encoder on low-light images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dirs = create_exp_dir(config['output_dir'], config['exp_name'])
    
    # Save config
    save_config(config, os.path.join(exp_dirs['logs'], 'config.yaml'))
    
    # Initialize TensorBoard
    writer = SummaryWriter(exp_dirs['logs'])
    
    # Create dataset and dataloader
    print("Loading datasets...")
    train_dataset = LowLightDataset(
        root_dir=config['dataset']['root_dir'],
        paired=False,  # Only use low-light images (unpaired mode)
        split='train',
        img_size=config['dataset']['img_size'],
        augment=True,
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize model
    print("Initializing MoCo v3 model...")
    model = MoCoV3Encoder(
        base_encoder=config['model']['base_encoder'],
        feat_dim=config['model']['feat_dim'],
        queue_size=config['model']['queue_size'],
        momentum=config['model']['momentum'],
        temperature=config['model']['temperature']
    ).to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr'],
        momentum=0.9,
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Starting training...")
    global_step = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get two augmented views of the same image
            im_q = batch['low'].to(device)
            
            # Create a second augmented view (in practice, use different augmentation)
            im_k = batch['low'].to(device)  # Simplified for demo
            
            # Forward pass
            logits, labels = model(im_q, im_k)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss.update(loss.item(), im_q.size(0))
            
            # Log to TensorBoard
            if global_step % 10 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{epoch_loss.avg:.4f}'})
            
            global_step += 1
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch metrics
        print(f"Epoch {epoch+1} - Average Loss: {epoch_loss.avg:.4f}")
        writer.add_scalar('Loss/epoch', epoch_loss.avg, epoch)
        
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