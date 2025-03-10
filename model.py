import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net architecture for wildfire spread prediction."""
    def __init__(self, n_channels: int, n_classes: int, feature_names: Optional[List[str]] = None):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.feature_names = feature_names  # Store feature names for reference
        
        # Log feature configuration
        if feature_names:
            print(f"Model initialized with {n_channels} channels: {', '.join(feature_names)}")
        else:
            print(f"Model initialized with {n_channels} channels")

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize model weights."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x4, x], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x3, x], dim=1))
        
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x2, x], dim=1))
        
        x = self.up4(x)
        x = self.conv_up4(torch.cat([x1, x], dim=1))
        
        # Output layer
        logits = self.outc(x)
        return torch.sigmoid(logits)
        
class FeatureImportanceAnalyzer:
    """Analyze and report feature importance for the wildfire model."""
    def __init__(self, model: nn.Module, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
    def analyze_gradient_based_importance(self, data_loader, device):
        """Analyze feature importance based on gradient magnitudes."""
        self.model.eval()
        feature_grads = {name: 0.0 for name in self.feature_names}
        samples_count = 0
        
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            inputs.requires_grad = True
            targets = targets.to(device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Calculate gradient magnitudes for each feature channel
            for i, name in enumerate(self.feature_names):
                # Take absolute mean of gradients for this feature channel
                grad_mag = inputs.grad[:, i].abs().mean().item()
                feature_grads[name] += grad_mag
            
            samples_count += 1
            
            # Limit analysis to first 100 batches
            if samples_count >= 100:
                break
                
        # Normalize by number of samples
        for name in feature_grads:
            feature_grads[name] /= samples_count
            
        # Sort features by importance
        sorted_features = sorted(feature_grads.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features