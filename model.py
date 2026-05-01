import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # For pretrained models (e.g., ResNet, EfficientNet)

class DocumentReconstructionModel(nn.Module):
    """
    Starter model for document dewarping (geometric correction).

    IMPORTANT: The goal is GEOMETRIC RECONSTRUCTION, not photometric matching!
    - The rendered images have lighting/shading effects
    - Your model should focus on learning the geometric transformation (UV/flow field)
    - Don't worry about exact pixel intensities - focus on structure

    TODO: Implement your own architecture here.
    This is a simple U-Net-style baseline to get started.

    Suggestions for improvement:
    - Use a pretrained encoder from HuggingFace (e.g., ResNet, EfficientNet)
    - Add attention mechanisms
    - Use depth/UV information if available
    - Experiment with different loss functions (SSIM is recommended!)
    - Add skip connections
    - Try different decoder architectures

    IMPORTANT HINT: Consider using torch.nn.functional.grid_sample for differentiable warping!

    One powerful approach for document reconstruction is to:
    1. Predict a deformation/flow field (mapping from distorted space to flat space)
    2. Use grid_sample to warp the input image according to this field
    3. This allows the network to learn geometric transformations explicitly

    Example usage of grid_sample:
        # Predict a flow field [B, 2, H, W] representing (x, y) offsets
        flow = self.flow_predictor(features)

        # Create base grid and add flow to get sampling coordinates
        grid = create_base_grid(B, H, W) + flow

        # Sample from input image using the predicted grid
        warped = torch.nn.functional.grid_sample(
            input_image,
            grid.permute(0, 2, 3, 1),  # [B, H, W, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        self.encoder = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=(0, 1, 2, 3, 4))

        # Decoder: upsample + concatenate skip connections
        self.dec4 = nn.Sequential(
            nn.Conv2d(2048 + 1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(32, 2, 1)  # 2 channels for flow field

        # TODO: Replace this simple architecture with your own design
        # Consider using HuggingFace transformers or timm models as backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Reconstructed image [B, 3, H, W]
            Flow field [B, 2, H, W]
        """
        # TODO: Implement your forward pass
        # Consider predicting a flow field and using grid_sample for warping!
        B, C, H, W = x.shape

        features = self.encoder(x)
        e0, e1, e2, e3, e4 = features  # ResNet-50 feature maps at different stages

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([
            F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=True),
            e3
        ], dim=1))

        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True),
            e2
        ], dim=1))

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True),
            e1
        ], dim=1))

        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e0.shape[2:], mode='bilinear', align_corners=True),
            e0
        ], dim=1))

        d0 = self.dec0(F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True))

        # Flow field
        flow = self.head(d0) # [B, 2, H, W]

        # return uv directly
        return flow
    
