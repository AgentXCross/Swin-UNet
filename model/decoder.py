import torch
from torch import nn

from model.basic_layer import BasicLayer
from model.expand_merge.patch_expanding import PatchExpanding


class Decoder(nn.Module):
    """
    Swin-UNet Decoder
    Structure:
        - Stage 1: PatchExpanding (1/32 -> 1/16), BasicLayer(depth3)
        - Stage 2: PatchExpanding (1/16 -> 1/8),  BasicLayer(depth2)
        - Stage 3: PatchExpanding (1/8  -> 1/4),  BasicLayer(depth1)
        - Final:   PatchExpanding (1/4  -> 1),    produces full-resolution features
    Input:
        x_bottleneck : (B, 8C, H/32, W/32)
        skips = [x_16, x_8, x_4]
    Output:
        (B, C/2, H, W)   # final feature map before segmentation head
    """
    def __init__(self, img_size, embed_dim, depths, num_heads, window_size = 7, mlp_ratio = 4.0):
        super().__init__()
        H, W = img_size, img_size
        # Decoder stage resolutions:
        # After up1: H/16, W/16
        # After up2: H/8,  W/8
        # After up3: H/4,  W/4
        # Final:     H,    W

        # 1/32 -> 1/16
        self.up1 = PatchExpanding(embed_dim * 8)
        self.stage1 = BasicLayer(
            dim = embed_dim * 4,
            input_resolution = (H // 16, W // 16),
            depth = depths[2],
            num_heads = num_heads[2],
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            upsample = False,
            downsample = False
        )
        # 1/16 -> 1/8
        self.up2 = PatchExpanding(embed_dim * 4)
        self.stage2 = BasicLayer(
            dim = embed_dim * 2,
            input_resolution = (H // 8, W // 8),
            depth = depths[1],
            num_heads = num_heads[1],
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            upsample = False,
            downsample = False
        )
        # 1/8 -> 1/4
        self.up3 = PatchExpanding(embed_dim * 2)
        self.stage3 = BasicLayer(
            dim = embed_dim,
            input_resolution = (H // 4, W // 4),
            depth = depths[0],
            num_heads = num_heads[0],
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            upsample = False,
            downsample = False
        )
        # 1/4 -> 1 (4x upsample)
        self.final_up = PatchExpanding(embed_dim)
    def forward(self, x, skips):
        # Unpack Skip Connections
        skip16, skip8, skip4 = skips
        # Stage 1 (1/32 -> 1/16)
        x = self.up1(x)
        x = x + skip16
        x = self.stage1(x)
        # Stage 2 (1/16 -> 1/8)
        x = self.up2(x)
        x = x + skip8
        x = self.stage2(x)
        # Stage 3 (1/8 -> 1/4)
        x = self.up3(x)
        x = x + skip4
        x = self.stage3(x)
        # Final 4x upsample: 1/4 -> 1
        x = self.final_up(x)
        return x
