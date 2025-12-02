import torch
from torch import nn

class PatchPartition(nn.Module):
    """
    Splits an input image into non-overlapping partitions.
    Input Shape: 
        (B, C, H, W)
    Operation:
        - Divides H and W into (H/p) * (W/p) grid of patches
        - Each patch is of size (C, p, p)
        - Flattens each patch into a vector of length (C * p * p)
    Output Shape:
        (B, H/p, W/p, C * p * p)
    First step of Swin-UNet. Converts the image into token-like patch embeddings
    before the linear embedding layer.
    Default Patch Size = 4
    """
    def __init__(self, patch_size = 4):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, H // p, W // p, C * p * p)
        return x

class LinearEmbedding(nn.Module):
    

