import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_loader import create_base_grid


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
