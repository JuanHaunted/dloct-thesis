"""
ConvNeXt U-Net for complex OCT B-scans (Re/Im channels in, Re/Im channels out).

No per-channel affine normalization that would treat Re and Im differently: GroupNorm(1, C)
normalizes over all channels jointly. The final conv is zero-initialized so a freshly built
network is the identity on its residual input.
"""

import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expansion, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-6))

    def forward(self, x):
        return x + self.gamma * self.pw2(self.act(self.pw1(self.norm(self.dwconv(x)))))


class Stage(nn.Sequential):
    def __init__(self, dim: int, depth: int):
        super().__init__(*[ConvNeXtBlock(dim) for _ in range(depth)])


class UNet(nn.Module):
    """
    Parameters
    ----------
    in_channels : 2 (Re, Im) + optional 1 mask channel of measured A-lines
    out_channels : 2 (Re, Im)
    base : channels at full resolution
    mults : channel multiplier per level; spatial size is halved between levels, so H and W
        must be divisible by 2 ** (len(mults) - 1)
    depth : ConvNeXt blocks per stage
    """

    def __init__(self, in_channels=3, out_channels=2, base=64, mults=(1, 2, 4, 8), depth=2):
        super().__init__()
        dims = [base * m for m in mults]
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        self.down_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, d in enumerate(dims[:-1]):
            self.down_stages.append(Stage(d, depth))
            self.downsamples.append(nn.Sequential(nn.GroupNorm(1, d), nn.Conv2d(d, dims[i + 1], 2, stride=2)))

        self.bottleneck = Stage(dims[-1], depth)

        self.upsamples = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.up_stages = nn.ModuleList()
        for i in reversed(range(len(dims) - 1)):
            self.upsamples.append(nn.ConvTranspose2d(dims[i + 1], dims[i], 2, stride=2))
            self.fuse.append(nn.Conv2d(2 * dims[i], dims[i], 1))
            self.up_stages.append(Stage(dims[i], depth))

        self.head = nn.Sequential(nn.GroupNorm(1, dims[0]), nn.Conv2d(dims[0], out_channels, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.divisor = 2 ** (len(dims) - 1)

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for stage, down in zip(self.down_stages, self.downsamples):
            h = stage(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)
        for up, fuse, stage in zip(self.upsamples, self.fuse, self.up_stages):
            h = up(h)
            h = stage(fuse(torch.cat([h, skips.pop()], dim=1)))
        return self.head(h)
