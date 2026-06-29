"""
A compact 3D Vision Transformer for volumetric classification.

Design goals
------------
* Plain PyTorch, no MONAI / FuseMedML dependency, so it is easy to read and to
  unit-test (see ``train.py`` and the synthetic-dot sanity experiment).
* Conv3d patch embedding (the standard ViT "patchify" implemented as a strided
  conv) -> a sequence of patch tokens -> a learnable [CLS] token -> a stack of
  Transformer encoder blocks -> linear head on the [CLS] token.
* Optional ``patch_padding_mask`` so the same model can later be used on
  variable-size tumour volumes that have been padded to a fixed divisible
  shape (mirrors the ``model.embed_mask_b`` idea in the GIST data pipeline).

Conventions
-----------
Input volumes are ``[B, C, Z, Y, X]`` (channel-first, matching the ``[Z, Y, X]``
array convention used by the data loader). ``img_size`` and ``patch_size`` are
given as ``(Z, Y, X)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _as_triple(v) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    z, y, x = v
    return (int(z), int(y), int(x))


class PatchEmbed3D(nn.Module):
    """Split a volume into non-overlapping patches and linearly embed each one.

    Implemented as a single strided Conv3d: kernel == stride == patch_size.
    Output: ``[B, num_patches, dim]`` plus the patch grid ``(gz, gy, gx)``.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        in_chans: int,
        dim: int,
    ):
        super().__init__()
        self.img_size = _as_triple(img_size)
        self.patch_size = _as_triple(patch_size)
        for s, p in zip(self.img_size, self.patch_size):
            if s % p != 0:
                raise ValueError(
                    f"img_size {self.img_size} not divisible by patch_size "
                    f"{self.patch_size}"
                )
        self.grid = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = self.grid[0] * self.grid[1] * self.grid[2]
        self.proj = nn.Conv3d(in_chans, dim, kernel_size=self.patch_size,
                              stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Z, Y, X] -> [B, dim, gz, gy, gx] -> [B, N, dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT3D(nn.Module):
    """A small 3D ViT classifier.

    Parameters
    ----------
    img_size, patch_size : (Z, Y, X)
    in_chans   : number of input channels (1 for a single MRI sequence;
                 set >1 to stack co-registered sequences as channels later).
    num_classes: output logits. Use 1 with BCEWithLogitsLoss for binary tasks
                 (the dot sanity experiment), or K for K-way CrossEntropy.
    dim, depth, heads, mlp_ratio : standard transformer width/depth knobs.
    """

    def __init__(
        self,
        img_size=(64, 64, 64),
        patch_size=(16, 16, 16),
        in_chans: int = 1,
        num_classes: int = 1,
        dim: int = 192,
        depth: int = 4,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, dim)
        n = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm: more stable to train from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        patch_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x : [B, C, Z, Y, X]
        patch_padding_mask : optional [B, num_patches] bool, True == PAD (ignored
            by attention). Mirrors ``model.embed_mask_b`` from the GIST pipeline,
            except there True meant "valid" -- here we follow PyTorch's
            convention where True means "masked out", so invert before passing.
        Returns logits [B, num_classes].
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)                     # [B, N, dim]
        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, dim]
        tokens = torch.cat([cls, tokens], dim=1)         # [B, N+1, dim]
        tokens = self.pos_drop(tokens + self.pos_embed)

        key_padding_mask = None
        if patch_padding_mask is not None:
            # CLS token is always valid -> prepend a False column.
            cls_col = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_col, patch_padding_mask], dim=1)

        tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        cls_out = self.norm(tokens[:, 0])                # [B, dim]
        return self.head(cls_out)                        # [B, num_classes]


if __name__ == "__main__":
    # Smoke test: shapes only (no training).
    model = ViT3D(img_size=(64, 64, 64), patch_size=(16, 16, 16),
                  num_classes=1, dim=192, depth=4, heads=3)
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output: {tuple(out.shape)}  params: {n_params/1e6:.2f}M  "
          f"patches: {model.patch_embed.num_patches}")
