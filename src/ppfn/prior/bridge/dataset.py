"""Same-domain bridge prior -- no warp, no registration.

A (scarce, restricted support) and B (abundant, uniform) are two point
clouds sampled from the SAME function `f` over the SAME `[0,1]^d` domain --
literally the ARCHITECTURE.md rho=0 case with the warp machinery removed
entirely, not merely forced to identity. Reuses the registration prior's
support-restriction logic (`sample_latent_A`/`sample_latent_B`, CLAUDE.md
invariant #7) and function prior (`sample_function_prior`), since that's the
one piece of the registration prior's design that matters even before
registration itself is turned on; skips `warp.py`/`normalize.py` entirely --
there is nothing to warp or normalize, x IS z.

Purpose: verify a plain encoder-decoder PFN's cross-attention pathway can
carry information from B into A's predictions at all, before any
registration machinery (warp/transport/coupling/affine) is layered on top --
one level below CLAUDE.md build-order step 4's go/no-go, which still carries
the full RegistrationPFN even at rho=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ppfn.prior.registration.function_prior import sample_function_prior
from ppfn.prior.registration.region import sample_latent_A, sample_latent_B

D_MAX = 5
D_CHOICES = (1, 2, 3, 5)


def build_bridge_item(
    rng: np.random.Generator,
    d: int | None = None,
    n_a_range: tuple[int, int] = (8, 100),
    n_b_range: tuple[int, int] = (8, 100),
) -> dict:
    """One (A, B) pair, same domain, no warp, no role randomization (A is
    always the decoder cloud, B always the encoder cloud -- role swapping is
    a registration-regime concern, ARCHITECTURE.md §4.3, not needed here).
    `d`/`n_a_range`/`n_b_range` mirror
    `ppfn.prior.registration.dataset.build_training_item`'s identical knobs."""
    if d is None:
        d = int(rng.choice(D_CHOICES))

    n_a = int(np.exp(rng.uniform(np.log(n_a_range[0]), np.log(n_a_range[1] + 1))))
    n_a = int(np.clip(n_a, *n_a_range))
    n_b = int(np.exp(rng.uniform(np.log(n_b_range[0]), np.log(n_b_range[1] + 1))))
    n_b = int(np.clip(n_b, *n_b_range))

    z_a, region, volume_fraction = sample_latent_A(rng, d, n_a)
    z_b = sample_latent_B(rng, d, n_b)

    f = sample_function_prior(rng, d, probe_z=np.concatenate([z_a, z_b], axis=0))
    y_a_raw = f.sample_y(rng, z_a)
    y_b_raw = f.sample_y(rng, z_b)

    y_pool = np.concatenate([y_a_raw, y_b_raw])
    y_mean, y_std = float(y_pool.mean()), max(float(y_pool.std()), 1e-6)
    y_a = (y_a_raw - y_mean) / y_std
    y_b = (y_b_raw - y_mean) / y_std

    k_ctx = int(rng.integers(1, max(2, n_a)))  # at least 1 context, at least 1 query
    k_ctx = min(k_ctx, n_a - 1) if n_a > 1 else 1
    perm = rng.permutation(n_a)
    ctx_idx, qry_idx = perm[:k_ctx], perm[k_ctx:]

    return {
        "d_real": d,
        "enc_x": z_b.astype(np.float32),
        "enc_y": y_b.astype(np.float32),
        "dec_ctx_x": z_a[ctx_idx].astype(np.float32),
        "dec_ctx_y": y_a[ctx_idx].astype(np.float32),
        "dec_qry_x": z_a[qry_idx].astype(np.float32),
        "y_qry": y_a[qry_idx].astype(np.float32),
        "region_type": region.kind,
        "volume_fraction": volume_fraction,
    }


@dataclass
class BridgeBatch:
    """Padded, batched tensors -- collate_bridge_batch's output. All
    point-count axes are padded to the batch max with zeros; `*_mask`
    tensors are True for real tokens, False for padding."""

    enc_x: torch.Tensor  # [B, n_enc, D_MAX]
    enc_y: torch.Tensor  # [B, n_enc]
    enc_mask: torch.Tensor  # [B, n_enc] bool
    dec_ctx_x: torch.Tensor  # [B, n_ctx, D_MAX]
    dec_ctx_y: torch.Tensor  # [B, n_ctx]
    dec_ctx_mask: torch.Tensor  # [B, n_ctx] bool
    dec_qry_x: torch.Tensor  # [B, n_qry, D_MAX]
    dec_qry_mask: torch.Tensor  # [B, n_qry] bool
    y_qry: torch.Tensor  # [B, n_qry]
    d_real: torch.Tensor  # [B] long


def _pad_stack(arrays: list[np.ndarray], max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """arrays: list of [n_i] or [n_i, F] (F fixed across the list). Returns
    (stacked [B, max_n] or [B, max_n, F], mask [B, max_n] bool, True=real)."""
    b = len(arrays)
    f = arrays[0].shape[-1] if arrays[0].ndim > 1 else None
    shape = (b, max_n, f) if f is not None else (b, max_n)
    out = np.zeros(shape, dtype=np.float32)
    mask = np.zeros((b, max_n), dtype=bool)
    for i, a in enumerate(arrays):
        n = a.shape[0]
        out[i, :n] = a
        mask[i, :n] = True
    return out, mask


def _pad_dim(x: np.ndarray, d_max: int = D_MAX) -> np.ndarray:
    """[N, d] -> [N, d_max], zero-padded columns -- no rescaling needed
    (unlike ppfn.model.pfn.pfn's VariableNumFeaturesEncoder-style fan-in
    correction), since d is batch-uniform here rather than truly per-item
    variable; a future variable-d-per-item run should switch to the
    rescale-then-pad convention instead."""
    n, d = x.shape
    if d == d_max:
        return x
    out = np.zeros((n, d_max), dtype=x.dtype)
    out[:, :d] = x
    return out


def collate_bridge_batch(items: list[dict]) -> BridgeBatch:
    n_enc = max(it["enc_x"].shape[0] for it in items)
    n_ctx = max(it["dec_ctx_x"].shape[0] for it in items)
    n_qry = max(it["dec_qry_x"].shape[0] for it in items)

    enc_x, enc_mask = _pad_stack([_pad_dim(it["enc_x"]) for it in items], n_enc)
    enc_y, _ = _pad_stack([it["enc_y"] for it in items], n_enc)
    dec_ctx_x, dec_ctx_mask = _pad_stack(
        [_pad_dim(it["dec_ctx_x"]) for it in items], n_ctx
    )
    dec_ctx_y, _ = _pad_stack([it["dec_ctx_y"] for it in items], n_ctx)
    dec_qry_x, dec_qry_mask = _pad_stack(
        [_pad_dim(it["dec_qry_x"]) for it in items], n_qry
    )
    y_qry, _ = _pad_stack([it["y_qry"] for it in items], n_qry)

    return BridgeBatch(
        enc_x=torch.from_numpy(enc_x),
        enc_y=torch.from_numpy(enc_y),
        enc_mask=torch.from_numpy(enc_mask),
        dec_ctx_x=torch.from_numpy(dec_ctx_x),
        dec_ctx_y=torch.from_numpy(dec_ctx_y),
        dec_ctx_mask=torch.from_numpy(dec_ctx_mask),
        dec_qry_x=torch.from_numpy(dec_qry_x),
        dec_qry_mask=torch.from_numpy(dec_qry_mask),
        y_qry=torch.from_numpy(y_qry),
        d_real=torch.tensor([it["d_real"] for it in items], dtype=torch.long),
    )


class BridgeStreamDataset(IterableDataset):
    """Infinite stream of `build_bridge_item` draws. One RNG per worker
    (seeded from `seed` + worker id) -- same idiom as
    `ppfn.prior.registration.dataset.RegistrationStreamDataset`, minus the
    rho-curriculum `SharedProgress` plumbing (there's no curriculum here)."""

    def __init__(
        self,
        seed: int = 0,
        d: int | None = None,
        n_a_range: tuple[int, int] = (8, 100),
        n_b_range: tuple[int, int] = (8, 100),
    ):
        super().__init__()
        self.seed = seed
        self.d = d
        self.n_a_range = n_a_range
        self.n_b_range = n_b_range

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id)
        while True:
            yield build_bridge_item(
                rng, d=self.d, n_a_range=self.n_a_range, n_b_range=self.n_b_range
            )


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: sample a batch,
    print shapes, and check the same-domain invariant directly -- A and B's
    coordinates must share literally the same [0,1]^d box (no warp box to
    even compute), unlike the registration prior's rho=0 check (which
    verifies the WARPED boxes happen to coincide)."""
    import numpy as np

    rng = np.random.default_rng(0)
    items = [build_bridge_item(rng, d=1) for _ in range(4)]
    batch = collate_bridge_batch(items)
    print("enc_x:", batch.enc_x.shape, "dec_ctx_x:", batch.dec_ctx_x.shape, "dec_qry_x:", batch.dec_qry_x.shape)
    print("d_real:", batch.d_real.tolist())

    all_x = np.concatenate(
        [it["enc_x"][:, 0] for it in items] + [it["dec_ctx_x"][:, 0] for it in items]
        + [it["dec_qry_x"][:, 0] for it in items]
    )
    print(f"pooled x range: [{all_x.min():.3f}, {all_x.max():.3f}] (expect within [0,1])")
