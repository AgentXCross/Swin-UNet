import torch
from torch import nn

from model.basic_layer import BasicLayer
from model.ops.patch_ops import PatchPartition, LinearEmbedding


class Encoder(nn.Module):
    """
    Swin-UNet Encoder.
    Structure:
        - Patch Partition (4x4)
        - Linear Embedding
        - Stage 1: BasicLayer(depth1)       skip at 1/4 resolution
        - Stage 2: BasicLayer(depth2)       skip at 1/8 resolution
        - Stage 3: BasicLayer(depth3)       skip at 1/16 resolution
        - Bottleneck: BasicLayer(depth4)    no downsampling
    Input Shape:
        (B, 3, H, W)
    Output:
        List of encoder outputs (skip connections), deepest first:
            [
                x_bottleneck,       # 1/32 resolution
                x_16,               # 1/16 resolution
                x_8,                # 1/8  resolution
                x_4                 # 1/4  resolution
            ]
    """
    def __init__(self, img_size, patch_size, embed_dim, depths, num_heads, window_size = 7, mlp_ratio = 4.0, in_channels = 3):
        super().__init__()
        H, W = img_size, img_size
        # Patch Partition
        self.patch_partition = PatchPartition(patch_size)
        # Linear Embedding into C = embed_dim
        patch_dim = in_channels * patch_size * patch_size
        self.linear_embed = LinearEmbedding(patch_dim, embed_dim)
        # Stage resolutions after partition:
        # Stage 1: H/4, W/4
        # Stage 2: H/8, W/8
        # Stage 3: H/16, W/16
        # Bottleneck: H/32, W/32
        self.stage1 = BasicLayer(
            dim = embed_dim,
            input_resolution = (H // 4, W // 4),
            depth = depths[0],
            num_heads = num_heads[0],
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            downsample = True
        )
        self.stage2 = BasicLayer(
            dim = embed_dim * 2,
            input_resolution = (H // 8, W // 8),
            depth = depths[1],
            num_heads = num_heads[1],
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            downsample = True
        )
        self.stage3 = BasicLayer(
            dim = embed_dim * 4,
            input_resolution = (H // 16, W // 16),
            depth = depths[2],
            num_heads = num_heads[2],
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            downsample = True
        )
        # Bottleneck (no downsample)
        self.bottleneck = BasicLayer(
            dim = embed_dim * 8,
            input_resolution = (H // 32, W // 32),
            depth = depths[3],
            num_heads = num_heads[3],
            window_size = window_size,
            mlp_ratio = mlp_ratio,
            downsample = False
        )
    def forward(self, x):
        # Partition patches into (B, H/4, W/4, 48)
        x = self.patch_partition(x)
        # Linear embedding -> (B, C, H/4, W/4)
        x = self.linear_embed(x)
        # Save skip connections in order: shallow -> deep
        skips = []
        # Stage 1 (1/4)
        x = self.stage1(x)
        skips.append(x)
        # Stage 2 (1/8)
        x = self.stage2(x)
        skips.append(x)
        # Stage 3 (1/16)
        x = self.stage3(x)
        skips.append(x)
        # Bottleneck (1/32)
        x = self.bottleneck(x)
        # Return: bottleneck first, then skips deepest -> shallow
        return [x, skips[2], skips[1], skips[0]]
