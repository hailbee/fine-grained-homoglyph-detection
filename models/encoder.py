"""Visual encoder: slice sequence → name embedding.

Architecture
------------
1. Flatten each slice from (height, slice_width) to a 1-D vector (slice_dim).
2. Two Conv1D layers (kernel_size=3, same padding) with ReLU, treating the
   slice sequence as a 1-D signal over the channel axis.
3. Pooling across the sequence dimension, selectable via the `pooling` parameter:
   - 'mean'      : global average pooling
   - 'max'       : global max pooling
   - 'attention' : single-layer attention — a linear layer scores each slice,
                   softmax normalises the scores, output is the weighted sum

Input shape:  (batch, num_slices, height, slice_width)
Output shape: (batch, embed_dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

POOLING_OPTIONS = ("mean", "max", "attention")


class VisualEncoder(nn.Module):
    """Encode a padded slice sequence into a fixed-size embedding.

    Args:
        slice_dim: Flattened size of one slice (height * slice_width).
        embed_dim: Size of the output embedding vector. Default: 128.
        pooling: Pooling strategy after the Conv1D stem — one of
            'mean', 'max', or 'attention'. Default: 'mean'.
    """

    def __init__(
        self,
        slice_dim: int,
        embed_dim: int = 128,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in POOLING_OPTIONS:
            raise ValueError(f"pooling must be one of {POOLING_OPTIONS}, got {pooling!r}")

        self.slice_dim = slice_dim
        self.embed_dim = embed_dim
        self.pooling = pooling

        # Conv1d expects (batch, channels, length).
        # We treat slice_dim as the channel axis and num_slices as length.
        self.conv = nn.Sequential(
            nn.Conv1d(slice_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Attention pooling: project each slice embedding to a scalar score.
        if pooling == "attention":
            self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, num_slices, height, slice_width)  — padded slice tensor

        Returns:
            (batch, embed_dim)
        """
        B, N, H, W = x.shape
        x = x.view(B, N, H * W)            # (B, num_slices, slice_dim)
        x = x.permute(0, 2, 1)             # (B, slice_dim, num_slices)
        x = self.conv(x)                    # (B, embed_dim, num_slices)

        if self.pooling == "mean":
            return x.mean(dim=2)            # (B, embed_dim)

        if self.pooling == "max":
            return x.max(dim=2).values      # (B, embed_dim)

        # attention
        x = x.permute(0, 2, 1)             # (B, num_slices, embed_dim)
        scores = self.attn_score(x)         # (B, num_slices, 1)
        weights = F.softmax(scores, dim=1)  # (B, num_slices, 1)
        return (weights * x).sum(dim=1)     # (B, embed_dim)


if __name__ == "__main__":
    height, slice_width, embed_dim = 32, 4, 128
    batch = torch.randn(4, 50, height, slice_width)

    for pooling in POOLING_OPTIONS:
        encoder = VisualEncoder(slice_dim=height * slice_width, embed_dim=embed_dim, pooling=pooling)
        out = encoder(batch)
        assert out.shape == (4, embed_dim), f"wrong shape for pooling={pooling!r}"
        print(f"pooling={pooling!r:12s}  input={tuple(batch.shape)}  output={tuple(out.shape)}")
