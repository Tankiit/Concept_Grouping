"""Development-only wrappers for real MNIST and a coloured spurious feature."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


_SEGMENTS = (
    "1111110", "0110000", "1101101", "1111001", "0110011",
    "1011011", "1011111", "1110000", "1111111", "1111011",
)


def _digit_masks(digits: torch.Tensor) -> torch.Tensor:
    """Render seven-segment digits into 28x28 binary images without downloads."""
    masks = torch.zeros((len(digits), 1, 28, 28), dtype=torch.float32)
    boxes = ((3, 6, 7, 21), (6, 13, 20, 23), (15, 22, 7, 21), (6, 13, 5, 9),
             (15, 22, 5, 9), (3, 6, 7, 21), (12, 16, 7, 21))
    for digit in range(10):
        rows = (digits == digit).nonzero(as_tuple=True)[0]
        for enabled, (r0, r1, c0, c1) in zip(_SEGMENTS[digit], boxes):
            if enabled == "1":
                masks[rows, 0, r0:r1, c0:c1] = 1.0
    return masks


class SyntheticColoredMNIST(Dataset):
    """Parity target with a configurable spurious colour correlation.

    This uses synthetic seven-segment glyphs, not the benchmark MNIST files;
    it is intentionally suitable only for checking the GPU/training path.
    """

    def __init__(self, n: int, rho: float, seed: int) -> None:
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1]")
        generator = torch.Generator().manual_seed(seed)
        digits = torch.randint(0, 10, (n,), generator=generator)
        parity = digits.remainder(2)
        colour = torch.where(
            torch.rand(n, generator=generator) < rho,
            parity,
            1 - parity,
        )
        mask = _digit_masks(digits)
        palette = torch.tensor([[0.95, 0.15, 0.15], [0.15, 0.35, 0.95]])
        image = mask * palette[colour][:, :, None, None]
        image = (image + 0.04 * torch.randn(image.shape, generator=generator)).clamp(0, 1)
        self.images, self.parity, self.colour = image, parity, colour

    def __len__(self) -> int:
        return len(self.parity)

    def __getitem__(self, index: int):
        return self.images[index], self.parity[index], torch.stack((self.parity[index], self.colour[index]))


class ColoredMNIST(Dataset):
    """Real MNIST digits with parity target and a controlled colour correlation.

    This is development-only data plumbing.  It must not be used as the
    preregistered DGP or certification implementation.
    """

    def __init__(self, root: str, *, train: bool, rho: float, seed: int, limit: int | None = None) -> None:
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1]")
        base = MNIST(root=root, train=train, download=False)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(base), generator=generator)
        if limit is not None:
            if not 0 < limit <= len(base):
                raise ValueError(f"limit must be in [1, {len(base)}]")
            indices = indices[:limit]
        self.pixels = base.data[indices].unsqueeze(1).float().div_(255)
        self.parity = base.targets[indices].remainder(2)
        self.colour = torch.where(
            torch.rand(len(indices), generator=generator) < rho,
            self.parity,
            1 - self.parity,
        )
        self.palette = torch.tensor([[0.95, 0.15, 0.15], [0.15, 0.35, 0.95]])

    def __len__(self) -> int:
        return len(self.parity)

    def __getitem__(self, index: int):
        image = self.pixels[index] * self.palette[self.colour[index]][:, None, None]
        return image, self.parity[index], torch.stack((self.parity[index], self.colour[index]))
