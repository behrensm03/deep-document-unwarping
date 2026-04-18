import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_loader import create_base_grid
import timm

class SimpleEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder: conv layers that downsample
        # decoder: upsample back to original size
        # head: final conv that outputs 2 channels (the flow field)

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)

        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(32, 2, kernel_size=1)  # output flow field (2 channels)

    def forward(self, x):
        # 1. pass through encoder-decoder to get flow [B, 2, H, W]
        # 2. create base grid
        # 3. add flow to base grid
        # 4. grid_sample to get rectified image
        # 5. return rectified image and flow
        B, C, H, W = x.size()

        # encoder
        z = self.enc1(x)
        z = self.enc2(self.pool(z))
        z = self.enc3(self.pool(z))

        # decoder
        z = self.dec3(F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=True))
        z = self.dec2(F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=True))
        z = self.dec1(z)

        # flow field
        flow = self.head(z)  # [B, 2, H, W]

        # warp input
        grid = create_base_grid(B, H, W, x.device)  # [B, H, W, 2]
        grid = grid + flow.permute(0, 2, 3, 1)  # add flow to base grid
        rectified = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)  # warp input

        return rectified, flow



class DocumentDewarpNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained ResNet-50 encoder, returns intermediate feature maps
        self.encoder = timm.create_model(
            'resnet50',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )

        # Encoder output channels at each stage
        # [64, 256, 512, 1024, 2048]

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

        # Head: 2 channel flow field output
        self.head = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Encoder: get feature maps at each stage
        features = self.encoder(x)
        e0, e1, e2, e3, e4 = features

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
        flow = torch.tanh(self.head(d0))  # [B, 2, H, W]

        # Warp input using predicted flow
        grid = create_base_grid(B, H, W, x.device)
        grid = grid + flow.permute(0, 2, 3, 1)
        rectified = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return rectified, flow
