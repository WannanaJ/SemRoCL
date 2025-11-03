"""
MoCo v3 Encoder for Unsupervised Contrastive Learning
Supports ResNet-18/50 backbone with momentum encoder and dynamic queue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy


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
                 queue_size=65536, momentum=0.999, temperature=0.07):
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
        
        # Build query encoder (online network)
        self.encoder_q = self._build_encoder(base_encoder, feat_dim)
        
        # Build key encoder (momentum network)
        self.encoder_k = self._build_encoder(base_encoder, feat_dim)
        
        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Key encoder is not trained by gradient
        
        # Create the queue for negative samples
        self.register_buffer("queue", torch.randn(feat_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _build_encoder(self, arch, feat_dim):
        """Build encoder backbone with projection head"""
        # Load pretrained ResNet
        if arch == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            base_feat_dim = 512
        elif arch == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            base_feat_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Remove final FC layer
        encoder = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add projection head (2-layer MLP)
        projection_head = nn.Sequential(
            nn.Linear(base_feat_dim, base_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(base_feat_dim, feat_dim)
        )
        
        return nn.Sequential(encoder, nn.Flatten(), projection_head)
    
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
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest keys in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
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
        q = self.encoder_q(im_q)  # (B, feat_dim)
        q = F.normalize(q, dim=1)
        
        if im_k is None:
            # Inference mode: return features only
            return q
        
        # Training mode: compute key features
        with torch.no_grad():
            # Update key encoder
            self._momentum_update_key_encoder()
            
            # Compute key features
            k = self.encoder_k(im_k)  # (B, feat_dim)
            k = F.normalize(k, dim=1)
        
        # Compute logits
        # Positive pairs: (B, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative pairs: (B, queue_size)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Concatenate logits: (B, 1 + queue_size)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positive is the first element
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return logits, labels
    
    def get_features(self, x, normalize=True):
        """Extract features for downstream tasks"""
        features = self.encoder_q(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features


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
