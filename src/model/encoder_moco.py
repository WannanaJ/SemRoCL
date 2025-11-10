"""
MoCo v3 Encoder for Unsupervised Contrastive Learning
Supports ResNet-18/50 backbone with momentum encoder and dynamic queue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
from typing import Dict, List, Tuple, Optional


class ResNetEncoder(nn.Module):
    """
    ResNet backbone wrapper that returns projection features and optional intermediate embeddings.
    """

    def __init__(
        self,
        arch: str,
        feat_dim: int,
        mid_layers: Optional[List[str]] = None,
    ):
        super().__init__()
        if arch == "resnet18":
            base_model = models.resnet18(pretrained=True)
            base_feat_dim = 512
        elif arch == "resnet50":
            base_model = models.resnet50(pretrained=True)
            base_feat_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.projection_head = nn.Sequential(
            nn.Linear(base_feat_dim, base_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_feat_dim, feat_dim),
        )

        self.mid_layers = mid_layers or []
        self.mid_heads = nn.ModuleDict()
        for name in self.mid_layers:
            if name not in {"layer1", "layer2", "layer3", "layer4"}:
                raise ValueError(f"Unsupported mid-layer: {name}")
            out_channels = getattr(self, name)[-1].conv3.out_channels if arch == "resnet50" else getattr(self, name)[-1].conv2.out_channels
            self.mid_heads[name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, feat_dim),
            )

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        feats: Dict[str, torch.Tensor] = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if return_intermediate and "layer1" in self.mid_layers:
            feats["layer1"] = F.normalize(self.mid_heads["layer1"](x), dim=1)

        x = self.layer2(x)
        if return_intermediate and "layer2" in self.mid_layers:
            feats["layer2"] = F.normalize(self.mid_heads["layer2"](x), dim=1)

        x = self.layer3(x)
        if return_intermediate and "layer3" in self.mid_layers:
            feats["layer3"] = F.normalize(self.mid_heads["layer3"](x), dim=1)

        x = self.layer4(x)
        if return_intermediate and "layer4" in self.mid_layers:
            feats["layer4"] = F.normalize(self.mid_heads["layer4"](x), dim=1)

        pooled = torch.flatten(self.avgpool(x), 1)
        proj = self.projection_head(pooled)
        proj = F.normalize(proj, dim=1)
        return proj, feats


class MoCoV3Encoder(nn.Module):
    """
    MoCo v3: Momentum Contrast for Unsupervised Visual Representation Learning v3
    
    Key features:
    - Query encoder (online network)
    - Key encoder (momentum network)
    - Dynamic queue for negative samples
    - Optional BYOL-style EMA update
    """
    
    def __init__(self, base_encoder='resnet18', feat_dim=128, 
                 queue_size=65536, momentum=0.999, temperature=0.07,
                 mid_layers: Optional[List[str]] = None):
        """
        Args:
            base_encoder: Backbone architecture ('resnet18' or 'resnet50')
            feat_dim: Dimension of output feature embeddings
            queue_size: Size of the queue for storing negative samples
            momentum: Momentum coefficient for updating key encoder
            temperature: Temperature parameter for InfoNCE loss
        """
        super(MoCoV3Encoder, self).__init__()
        
        self.feat_dim = feat_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.base_temperature = temperature
        self.mid_layers = mid_layers or []
        
        # Build query encoder (online network)
        self.encoder_q = ResNetEncoder(base_encoder, feat_dim, self.mid_layers)
        
        # Build key encoder (momentum network)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Create the queue for negative samples
        self.queue: torch.Tensor
        self.queue_ptr: torch.Tensor
        self.register_buffer("queue", torch.randn(feat_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.last_pos_sim = None
        self.last_neg_logits = None
        self.last_q = None
        self.last_k = None
        self.last_mid_q = {}
        self.last_mid_k = {}
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue with new keys"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr.item())
        
        # Replace oldest keys in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = torch.tensor(ptr, device=self.queue_ptr.device, dtype=self.queue_ptr.dtype)
    
    def forward(self, im_q, im_k=None):
        """
        Forward pass for MoCo v3
        
        Args:
            im_q: Query images (B, 3, H, W)
            im_k: Key images (B, 3, H, W) - optional, for training only
        
        Returns:
            If training (im_k provided): logits, labels
            If inference: query features
        """
        # Compute query features
        q, q_feats = self.encoder_q(im_q, return_intermediate=bool(self.mid_layers))
        self.last_q = q.detach()
        self.last_mid_q = {name: feat.detach() for name, feat in q_feats.items()}
        
        if im_k is None:
            # Inference mode: return features only
            self.last_k = None
            self.last_pos_sim = None
            self.last_neg_logits = None
            self.last_mid_k = None
            return q
        
        # Training mode: compute key features
        with torch.no_grad():
            # Update key encoder
            self._momentum_update_key_encoder()
            
            # Compute key features
            k, k_feats = self.encoder_k(im_k, return_intermediate=bool(self.mid_layers))
            self.last_k = k.detach()
            self.last_mid_k = {name: feat.detach() for name, feat in k_feats.items()}
        
        # Compute logits
        # Positive pairs: (B, 1)
        pos_sim = torch.einsum('nc,nc->n', [q, k])
        l_pos = pos_sim.unsqueeze(-1)
        # Negative pairs: (B, queue_size)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        self.last_pos_sim = pos_sim.detach()
        self.last_neg_logits = l_neg.detach()
        
        # Concatenate logits: (B, 1 + queue_size)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positive is the first element
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels, {"mid_q": q_feats, "mid_k": k_feats}
    
    def get_features(self, x, normalize=True):
        """Extract features for downstream tasks"""
        features, _ = self.encoder_q(x, return_intermediate=False)
        if normalize:
            features = F.normalize(features, dim=1)
        return features
    
    def set_temperature(self, tau: float):
        """Dynamically update temperature for contrastive logits."""
        self.temperature = float(tau)


class BYOLEncoder(nn.Module):
    """
    BYOL-style encoder as an alternative to MoCo
    Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
    """
    
    def __init__(self, base_encoder='resnet18', feat_dim=128, 
                 hidden_dim=4096, momentum=0.996):
        super(BYOLEncoder, self).__init__()
        
        self.momentum = momentum
        
        # Online network
        self.online_encoder = self._build_encoder(base_encoder, hidden_dim, feat_dim)
        # Predictor (only for online network)
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim)
        )
        
        # Target network (EMA of online network)
        self.target_encoder = self._build_encoder(base_encoder, hidden_dim, feat_dim)
        
        # Initialize target network
        for param_o, param_t in zip(self.online_encoder.parameters(),
                                     self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
    
    def _build_encoder(self, arch, hidden_dim, feat_dim):
        """Build encoder with projector"""
        if arch == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            base_feat_dim = 512
        elif arch == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            base_feat_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        encoder = nn.Sequential(*list(base_model.children())[:-1])
        projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim)
        )
        
        return nn.Sequential(encoder, projector)
    
    @torch.no_grad()
    def update_target_network(self):
        """EMA update of target network"""
        for param_o, param_t in zip(self.online_encoder.parameters(),
                                     self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)
    
    def forward(self, x1, x2):
        """
        Forward pass for BYOL
        
        Args:
            x1, x2: Two augmented views of the same image
        
        Returns:
            Predicted and target features for loss computation
        """
        # Online network predictions
        online_proj_1 = self.online_encoder(x1)
        online_proj_2 = self.online_encoder(x2)
        
        online_pred_1 = self.predictor(online_proj_1)
        online_pred_2 = self.predictor(online_proj_2)
        
        # Target network projections
        with torch.no_grad():
            target_proj_1 = self.target_encoder(x1)
            target_proj_2 = self.target_encoder(x2)
        
        return online_pred_1, online_pred_2, target_proj_1, target_proj_2


if __name__ == '__main__':
    # Test MoCo encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MoCoV3Encoder(base_encoder='resnet18', feat_dim=128).to(device)
    
    # Test training mode
    im_q = torch.randn(8, 3, 224, 224).to(device)
    im_k = torch.randn(8, 3, 224, 224).to(device)
    
    logits, labels = model(im_q, im_k)
    print(f"Training mode - Logits shape: {logits.shape}, Labels shape: {labels.shape}")
    
    # Test inference mode
    features = model(im_q)
    print(f"Inference mode - Features shape: {features.shape}")
