"""
Lightweight Semantic Segmentation Head with Uncertainty Estimation
Uses SegFormer-B0 or MobileViT-UNet for efficient semantic guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import timm


class SemanticHead(nn.Module):
    """
    Semantic segmentation head with uncertainty estimation
    
    Provides:
    - Semantic segmentation masks
    - Uncertainty/confidence maps for dynamic weighting
    """
    
    def __init__(self, num_classes=19, backbone='segformer-b0', 
                 pretrained=True, uncertainty_estimation=True):
        """
        Args:
            num_classes: Number of semantic classes (19 for Cityscapes)
            backbone: 'segformer-b0', 'segformer-b1', or 'mobilevit'
            pretrained: Load pretrained weights
            uncertainty_estimation: Enable uncertainty estimation branch
        """
        super(SemanticHead, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.uncertainty_estimation = uncertainty_estimation
        
        # Build backbone
        if 'segformer' in backbone:
            self.backbone = self._build_segformer(backbone, num_classes, pretrained)
        elif 'mobilevit' in backbone:
            self.backbone = self._build_mobilevit(num_classes, pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Uncertainty estimation head (if enabled)
        if self.uncertainty_estimation:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()  # Output confidence in [0, 1]
            )
    
    def _build_segformer(self, model_name, num_classes, pretrained):
        """Build SegFormer model"""
        if model_name == 'segformer-b0':
            model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        elif model_name == 'segformer-b1':
            model_id = "nvidia/segformer-b1-finetuned-ade-512-512"
        else:
            model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            model = SegformerForSemanticSegmentation(config)
        
        return model
    
    def _build_mobilevit(self, num_classes, pretrained):
        """Build MobileViT-based UNet"""
        # Use timm's mobilevit as encoder
        encoder = timm.create_model(
            'mobilevit_s',
            pretrained=pretrained,
            features_only=True
        )
        
        # Simple decoder for semantic segmentation
        decoder = SimpleDecoder(
            encoder_channels=[32, 64, 96, 128, 160],
            num_classes=num_classes
        )
        
        return nn.ModuleDict({'encoder': encoder, 'decoder': decoder})
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            seg_logits: Segmentation logits (B, num_classes, H, W)
            confidence: Uncertainty/confidence map (B, 1, H, W) if enabled
        """
        B, C, H, W = x.shape
        
        # Get segmentation logits
        if 'segformer' in self.backbone_name:
            outputs = self.backbone(x)
            seg_logits = outputs.logits
            
            # Upsample to original size
            seg_logits = F.interpolate(
                seg_logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        else:
            # MobileViT encoder-decoder
            features = self.backbone['encoder'](x)
            seg_logits = self.backbone['decoder'](features)
            seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
        
        # Compute uncertainty/confidence map
        if self.uncertainty_estimation:
            # Compute confidence based on prediction entropy
            seg_probs = F.softmax(seg_logits, dim=1)
            confidence = self.uncertainty_head(seg_logits)
            
            # Alternative: Use prediction entropy as uncertainty
            # entropy = -torch.sum(seg_probs * torch.log(seg_probs + 1e-10), dim=1, keepdim=True)
            # confidence = 1 - (entropy / torch.log(torch.tensor(self.num_classes)))
            
            return seg_logits, confidence
        else:
            return seg_logits, None
    
    def get_semantic_features(self, x):
        """
        Extract multi-scale semantic features for enhancement
        
        Returns a list of feature maps at different scales.
        For SegFormer-B0, returns 4 hidden states with channels [64, 128, 320, 512]
        """
        if 'segformer' in self.backbone_name:
            # Extract intermediate features from SegFormer
            # IMPORTANT: Use the segformer encoder, not the full model
            outputs = self.backbone.segformer(
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True
            )
            
            # hidden_states is a tuple of (B, num_channels, H, W) tensors
            # For SegFormer-B0: [64, 128, 320, 512] channels
            features = outputs.hidden_states
            
            # Debug: Print shapes to verify
            # print(f"Number of hidden states: {len(features)}")
            # for i, feat in enumerate(features):
            #     print(f"  Hidden state {i}: {feat.shape}")
            
            return features
        else:
            # Extract features from MobileViT encoder
            features = self.backbone['encoder'](x)
            return features


class SimpleDecoder(nn.Module):
    """Simple decoder for MobileViT encoder"""
    
    def __init__(self, encoder_channels, num_classes):
        super(SimpleDecoder, self).__init__()
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        channels = encoder_channels[::-1]  # Reverse order
        
        for i in range(len(channels) - 1):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i] + channels[i+1], channels[i+1], 3, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], 3, padding=1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(channels[-1], num_classes, 1)
    
    def forward(self, features):
        """
        Args:
            features: List of encoder features (from low to high resolution)
        
        Returns:
            Segmentation logits
        """
        features = features[::-1]  # Reverse to start from deepest
        
        x = features[0]
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample and concatenate with skip connection
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, features[i+1]], dim=1)
            x = decoder_block(x)
        
        # Final segmentation
        seg_logits = self.seg_head(x)
        
        return seg_logits


class UIoUUncertainty(nn.Module):
    """
    Uncertainty estimation using Intersection over Union (UIoU)
    Based on prediction consistency
    """
    
    def __init__(self, num_iterations=5):
        super(UIoUUncertainty, self).__init__()
        self.num_iterations = num_iterations
    
    def forward(self, model, x):
        """
        Estimate uncertainty through multiple forward passes with dropout
        
        Args:
            model: Semantic segmentation model
            x: Input image
        
        Returns:
            Uncertainty map based on prediction variance
        """
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.num_iterations):
            with torch.no_grad():
                seg_logits, _ = model(x)
                seg_probs = F.softmax(seg_logits, dim=1)
                predictions.append(seg_probs)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_iterations, B, C, H, W)
        
        # Compute variance across predictions
        mean_pred = predictions.mean(dim=0)
        variance = ((predictions - mean_pred.unsqueeze(0)) ** 2).mean(dim=0)
        
        # Aggregate variance across channels
        uncertainty = variance.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Convert to confidence (inverse of uncertainty)
        confidence = 1 - (uncertainty / (uncertainty.max() + 1e-10))
        
        model.eval()
        
        return confidence


if __name__ == '__main__':
    # Test semantic head
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SemanticHead(
        num_classes=19,
        backbone='segformer-b0',
        pretrained=False,
        uncertainty_estimation=True
    ).to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512).to(device)
    seg_logits, confidence = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Segmentation logits shape: {seg_logits.shape}")
    print(f"Confidence map shape: {confidence.shape if confidence is not None else None}")
    
    # Test feature extraction - THIS IS THE KEY TEST
    print("\n" + "="*60)
    print("Testing get_semantic_features():")
    print("="*60)
    features = model.get_semantic_features(x)
    print(f"Number of feature levels: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Feature level {i}: {feat.shape}")
    
    # Verify the first feature has 64 channels (for SegFormer-B0)
    if features[0].shape[1] == 64:
        print("\n✅ CORRECT: First feature has 64 channels (matches SegFormer-B0)")
    else:
        print(f"\n❌ ERROR: First feature has {features[0].shape[1]} channels, expected 64")