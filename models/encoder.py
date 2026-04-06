"""Visual encoder: slice sequence → name embedding.

Architecture
------------
1. Flatten each slice from (height, slice_width) to a 1-D vector (slice_dim).
2. Two Conv1D layers (kernel_size=3, same padding) with ReLU, treating the
   slice sequence as a 1-D signal over the channel axis.
3. Mean-pool across the sequence to produce a fixed-size embedding.

Input shape:  (batch, num_slices, height, slice_width)
Output shape: (batch, embed_dim)
"""
import torch
import torch.nn as nn
from torch import Tensor


class VisualEncoder(nn.Module):
    """Encode a padded slice sequence into a fixed-size embedding.

    Args:
        slice_dim: Flattened size of one slice (height * slice_width).
        embed_dim: Size of the output embedding vector. Default: 128.
    """

    def __init__(self, slice_dim: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.slice_dim = slice_dim
        self.embed_dim = embed_dim

        # Conv1d expects (batch, channels, length).
        # We treat slice_dim as the channel axis and num_slices as length.
        self.conv = nn.Sequential(
            nn.Conv1d(slice_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, num_slices, height, slice_width)  — padded slice tensor

        Returns:
            (batch, embed_dim)
        """
        B, N, H, W = x.shape
        # Flatten each slice into a 1-D vector
        x = x.view(B, N, H * W)           # (B, num_slices, slice_dim)
        x = x.permute(0, 2, 1)            # (B, slice_dim, num_slices)
        x = self.conv(x)                   # (B, embed_dim, num_slices)
        x = x.mean(dim=2)                  # (B, embed_dim)  — mean pool
        return x


if __name__ == "__main__":
    height, slice_width, embed_dim = 32, 4, 128
    encoder = VisualEncoder(slice_dim=height * slice_width, embed_dim=embed_dim)
    print(encoder)

    batch = torch.zeros(4, 50, height, slice_width)
    out = encoder(batch)
    print(f"input={tuple(batch.shape)}  output={tuple(out.shape)}")
    assert out.shape == (4, embed_dim)
