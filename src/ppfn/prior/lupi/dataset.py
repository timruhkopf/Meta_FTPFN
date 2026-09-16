"""torch-facing wrapper around `sampler.sample_pair` -- the numpy/torch
boundary CLAUDE.md's repo conventions call for. Mirrors
`ppfn.prior.registration.dataset` structurally (SharedProgress, an
IterableDataset yielding `build_training_item` draws, a padded/masked batch
dataclass + collate_fn) but is NOT a subclass/reuse of it: the item shape is
different (an `oracle_bpos` channel per A token instead of a `transport`
channel, no `severed`/`role_swapped` flags -- this model doesn't do role
randomization or a severed-mode gate, see `docs/labbook/`'s LUPI build spec).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ppfn.prior.lupi.sampler import D_CHOICES, sample_pair, sample_rho_curriculum

D_MAX = max(D_CHOICES)


class SharedProgress:
    """Identical device to `ppfn.prior.registration.dataset.SharedProgress`
    -- see that module's docstring for the `multiprocessing.Value` rationale.
    Duplicated rather than imported so this package doesn't reach into the
    registration prior's dataset module for anything beyond the warp/region/
    function_prior pieces `sampler.py` already reuses."""

    def __init__(self, initial: float = 0.0):
        self._value = mp.Value("d", initial)

    def get(self) -> float:
        return float(self._value.value)

    def set(self, progress: float) -> None:
        self._value.value = float(progress)


def build_training_item(
    rng: np.random.Generator,
    progress: float,
    s_max: float = 0.1,
    force_rho_zero: bool = False,
    d: int | None = None,
    n_a_range: tuple[int, int] = (8, 256),
    n_b_range: tuple[int, int] = (256, 1024),
    n_qry_range: tuple[int, int] = (8, 128),
    frac_uniform: float = 0.4,
    frac_near_b: float = 0.4,
    query_eps_std: float = 0.03,
    warp_grid_n: int = 4,
    beta_override: float | None = None,
) -> dict:
    rho = 0.0 if force_rho_zero else sample_rho_curriculum(rng, progress)
    pair = sample_pair(
        rng, rho=rho, s_max=s_max, d=d, n_a_range=n_a_range, n_b_range=n_b_range,
        n_qry_range=n_qry_range, frac_uniform=frac_uniform, frac_near_b=frac_near_b,
        query_eps_std=query_eps_std, warp_grid_n=warp_grid_n, beta_override=beta_override,
    )
    return {
        "d_real": pair.d,
        "rho": np.float32(pair.rho),
        "beta": np.float32(pair.beta),
        "enc_x": pair.x_b.astype(np.float32),
        "enc_z": pair.z_b.astype(np.float32),
        "enc_x_inA": pair.x_b_inA.astype(np.float32),  # A-frame position, same n_b as enc_x
        "enc_z_inA": pair.z_b_inA.astype(np.float32),  # A-frame VALUE (h applied) -- pair with enc_x_inA, NOT enc_z, for a fully-registered oracle view (see LUPIPair.z_b_inA)
        "dec_ctx_x": pair.x_a_ctx.astype(np.float32),
        "dec_ctx_z": pair.z_a_ctx.astype(np.float32),
        "dec_ctx_oracle_bpos": pair.oracle_bpos_a_ctx.astype(np.float32),
        "dec_qry_x": pair.x_a_qry.astype(np.float32),
        "dec_qry_z": pair.z_a_qry.astype(np.float32),
        "dec_qry_oracle_bpos": pair.oracle_bpos_a_qry.astype(np.float32),
        "dec_qry_source": pair.qry_source,  # int8: 0=uniform, 1=near-B, 2=near-A-context
        "region_type": pair.meta["region_type"],
        "volume_fraction": pair.meta["volume_fraction"],
    }


class LUPIStreamDataset(IterableDataset):
    """Infinite stream of `build_training_item` draws -- see
    `ppfn.prior.registration.dataset.RegistrationStreamDataset`'s docstring
    for the per-worker-RNG rationale, identical here."""

    def __init__(
        self,
        seed: int = 0,
        s_max: float = 0.1,
        force_rho_zero: bool = False,
        progress: SharedProgress | None = None,
        d: int | None = None,
        n_a_range: tuple[int, int] = (8, 256),
        n_b_range: tuple[int, int] = (256, 1024),
        n_qry_range: tuple[int, int] = (8, 128),
        frac_uniform: float = 0.4,
        frac_near_b: float = 0.4,
        query_eps_std: float = 0.03,
        warp_grid_n: int = 4,
        beta_override: float | None = None,
    ):
        super().__init__()
        self.seed = seed
        self.s_max = s_max
        self.force_rho_zero = force_rho_zero
        self.progress = progress or SharedProgress(0.0)
        self.d = d
        self.n_a_range = n_a_range
        self.n_b_range = n_b_range
        self.n_qry_range = n_qry_range
        self.frac_uniform = frac_uniform
        self.frac_near_b = frac_near_b
        self.query_eps_std = query_eps_std
        self.warp_grid_n = warp_grid_n
        self.beta_override = beta_override

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id)
        while True:
            yield build_training_item(
                rng,
                progress=self.progress.get(),
                s_max=self.s_max,
                force_rho_zero=self.force_rho_zero,
                d=self.d,
                n_a_range=self.n_a_range,
                n_b_range=self.n_b_range,
                n_qry_range=self.n_qry_range,
                frac_uniform=self.frac_uniform,
                frac_near_b=self.frac_near_b,
                query_eps_std=self.query_eps_std,
                warp_grid_n=self.warp_grid_n,
                beta_override=self.beta_override,
            )


def _pad_rescale_coords(x: np.ndarray, d_real: int, d_max: int = D_MAX) -> np.ndarray:
    """Identical fan-in-correction trick to
    `ppfn.prior.registration.dataset._pad_rescale_coords` -- duplicated (not
    imported) since that function is a module-private helper of a different
    package; see that module's docstring for why the rescale is needed at
    all (a fixed-width Linear layer, ragged real dimensionality)."""
    n = x.shape[0]
    scaled = x * (d_max / d_real)
    if d_real == d_max:
        return scaled
    return np.concatenate([scaled, np.zeros((n, d_max - d_real), dtype=x.dtype)], axis=-1)


@dataclass
class LUPIBatch:
    """Padded, batched tensors -- collate_fn's output. `*_mask`: True = real
    token, False = padding. `dec_ctx_oracle_bpos`/`dec_qry_oracle_bpos`: the
    ground-truth B-frame position T(x) -- injected as the coordinate input
    in oracle mode (`ppfn.model.lupi.model.LUPIPFN.align(..., mode="oracle")`),
    ignored entirely in student mode."""

    enc_x: torch.Tensor  # [B, n_enc, D_MAX]
    enc_z: torch.Tensor  # [B, n_enc]
    enc_x_inA: torch.Tensor  # [B, n_enc, D_MAX]  B transported into A's frame via T
    enc_z_inA: torch.Tensor  # [B, n_enc]  B's value recalibrated into A's SCALE via h --
    # pair with enc_x_inA (NOT enc_z) for a fully-registered oracle view: T
    # and h both solved, so (enc_x_inA, enc_z_inA) sits on A's own true
    # curve up to B's own observation noise. `enc_z` is deliberately kept
    # separate -- it's B's raw value on B's own scale, for the student's
    # raw/unregistered view and for a "T solved, h still not" ablation.
    enc_mask: torch.Tensor  # [B, n_enc] bool

    dec_ctx_x: torch.Tensor  # [B, n_ctx, D_MAX]
    dec_ctx_z: torch.Tensor  # [B, n_ctx]
    dec_ctx_oracle_bpos: torch.Tensor  # [B, n_ctx, D_MAX]
    dec_ctx_mask: torch.Tensor  # [B, n_ctx] bool

    dec_qry_x: torch.Tensor  # [B, n_qry, D_MAX]
    dec_qry_z: torch.Tensor  # [B, n_qry]  target
    dec_qry_oracle_bpos: torch.Tensor  # [B, n_qry, D_MAX]
    dec_qry_mask: torch.Tensor  # [B, n_qry] bool
    dec_qry_source: torch.Tensor  # [B, n_qry] long: 0=uniform, 1=near-B, 2=near-A-context (padding=0, masked out)

    d_real: torch.Tensor  # [B] long
    rho: torch.Tensor  # [B] float
    beta: torch.Tensor  # [B] float
    meta: list  # list[dict], length B, un-batched


def _pad_stack(arrays: list[np.ndarray], max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """arrays: list of [n_i, F] or [n_i]. Returns (stacked [B, max_n, (F)],
    mask [B, max_n] bool, True=real) -- identical convention to
    `ppfn.prior.registration.dataset._pad_stack`, duplicated for the same
    module-private-helper reason as `_pad_rescale_coords` above."""
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


def collate_lupi_batch(items: list[dict]) -> LUPIBatch:
    padded = [
        {
            **it,
            "enc_x": _pad_rescale_coords(it["enc_x"], it["d_real"]),
            "enc_x_inA": _pad_rescale_coords(it["enc_x_inA"], it["d_real"]),
            "dec_ctx_x": _pad_rescale_coords(it["dec_ctx_x"], it["d_real"]),
            "dec_ctx_oracle_bpos": _pad_rescale_coords(it["dec_ctx_oracle_bpos"], it["d_real"]),
            "dec_qry_x": _pad_rescale_coords(it["dec_qry_x"], it["d_real"]),
            "dec_qry_oracle_bpos": _pad_rescale_coords(it["dec_qry_oracle_bpos"], it["d_real"]),
        }
        for it in items
    ]

    max_n_enc = max(it["enc_x"].shape[0] for it in padded)
    max_n_ctx = max(it["dec_ctx_x"].shape[0] for it in padded)
    max_n_qry = max(it["dec_qry_x"].shape[0] for it in padded)

    enc_x, enc_mask = _pad_stack([it["enc_x"] for it in padded], max_n_enc)
    enc_z, _ = _pad_stack([it["enc_z"] for it in padded], max_n_enc)
    enc_x_inA, _ = _pad_stack([it["enc_x_inA"] for it in padded], max_n_enc)
    enc_z_inA, _ = _pad_stack([it["enc_z_inA"] for it in padded], max_n_enc)
    dec_ctx_x, dec_ctx_mask = _pad_stack([it["dec_ctx_x"] for it in padded], max_n_ctx)
    dec_ctx_z, _ = _pad_stack([it["dec_ctx_z"] for it in padded], max_n_ctx)
    dec_ctx_oracle_bpos, _ = _pad_stack([it["dec_ctx_oracle_bpos"] for it in padded], max_n_ctx)
    dec_qry_x, dec_qry_mask = _pad_stack([it["dec_qry_x"] for it in padded], max_n_qry)
    dec_qry_z, _ = _pad_stack([it["dec_qry_z"] for it in padded], max_n_qry)
    dec_qry_oracle_bpos, _ = _pad_stack([it["dec_qry_oracle_bpos"] for it in padded], max_n_qry)
    dec_qry_source, _ = _pad_stack(
        [it["dec_qry_source"].astype(np.float32) for it in padded], max_n_qry
    )

    return LUPIBatch(
        enc_x=torch.from_numpy(enc_x),
        enc_z=torch.from_numpy(enc_z),
        enc_x_inA=torch.from_numpy(enc_x_inA),
        enc_z_inA=torch.from_numpy(enc_z_inA),
        enc_mask=torch.from_numpy(enc_mask),
        dec_ctx_x=torch.from_numpy(dec_ctx_x),
        dec_ctx_z=torch.from_numpy(dec_ctx_z),
        dec_ctx_oracle_bpos=torch.from_numpy(dec_ctx_oracle_bpos),
        dec_ctx_mask=torch.from_numpy(dec_ctx_mask),
        dec_qry_x=torch.from_numpy(dec_qry_x),
        dec_qry_z=torch.from_numpy(dec_qry_z),
        dec_qry_oracle_bpos=torch.from_numpy(dec_qry_oracle_bpos),
        dec_qry_mask=torch.from_numpy(dec_qry_mask),
        dec_qry_source=torch.from_numpy(dec_qry_source).long(),
        d_real=torch.tensor([it["d_real"] for it in items], dtype=torch.long),
        rho=torch.tensor([it["rho"] for it in items], dtype=torch.float32),
        beta=torch.tensor([it["beta"] for it in items], dtype=torch.float32),
        meta=[
            {"region_type": it["region_type"], "volume_fraction": it["volume_fraction"]}
            for it in items
        ],
    )


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build one batch,
    print every tensor's shape/dtype, and confirm the rho=0 invariant
    survives padding/collation (oracle position == A-frame position on the
    real, non-padded entries)."""
    from torch.utils.data import DataLoader

    dataset = LUPIStreamDataset(seed=0, s_max=0.1, force_rho_zero=True)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_lupi_batch)
    batch = next(iter(loader))
    for f in fields(LUPIBatch):
        if f.name == "meta":
            continue
        t = getattr(batch, f.name)
        print(f"{f.name:24s} {tuple(t.shape)} {t.dtype}")

    ctx_diff = (batch.dec_ctx_oracle_bpos - batch.dec_ctx_x)[batch.dec_ctx_mask].abs().max()
    qry_diff = (batch.dec_qry_oracle_bpos - batch.dec_qry_x)[batch.dec_qry_mask].abs().max()
    print("rho=0 check, ctx |oracle_bpos - x| max (expect ~0):", ctx_diff.item())
    print("rho=0 check, qry |oracle_bpos - x| max (expect ~0):", qry_diff.item())
