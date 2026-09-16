"""torch-facing wrapper around `sampler.sample_pair` -- the numpy/torch
boundary CLAUDE.md's repo conventions call for ("Prior in numpy, model in
torch. The prior runs in dataloader workers.").

Per-item responsibilities living HERE, not in `sampler.py`:
  * role randomization (ARCHITECTURE.md §4.3): w.p. 0.35, swap which cloud
    is "encoder" and which is "decoder". `sample_pair` always returns a
    symmetric pair with BOTH pushforward targets (A_inB_target,
    B_inA_target) computed regardless, so the swap is pure relabeling here,
    no extra generative work.
  * the decoder-side context/query split (ARCHITECTURE.md §2's "a PFN over
    the scarce cloud A plus its query points") -- a masking decision over
    the decoder cloud's flat point set, resampled every draw (standard PFN
    single_eval_pos convention), not part of the generative model.
  * the severed-mode coin flip (ARCHITECTURE.md §4.4) -- carried as a
    per-item flag (`severed`) rather than a per-batch one, since the gate
    application in `ppfn.model.registration` multiplies each item's
    cross-attention contribution by its own `severed` mask, so a batch can
    freely mix full/severed items (statistically more efficient than
    forcing one mode per whole batch, and no harder to implement).

`rho` is drawn from `sampler.sample_rho_curriculum`, which needs training
PROGRESS (fraction of steps completed) -- state a dataloader worker process
can't see on its own. `SharedProgress` is a `multiprocessing.Value` the
training loop updates every step from the main process; under the default
'fork' start method (Linux) it stays valid in already-spawned worker
processes because it's backed by shared memory, not copied at fork time.
`num_workers=0` (the shipped default, see `configs/prior/registration.yaml`)
sidesteps this entirely -- everything runs in the main process -- and is
the recommended starting point; bump `num_workers` once multi-process
behavior has been checked on the target machine.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ppfn.prior.registration.sampler import (
    D_CHOICES,
    sample_pair,
    sample_rho_curriculum,
)

D_MAX = max(D_CHOICES)


class SharedProgress:
    """A `multiprocessing.Value('d', 0.0)` wrapper -- see module docstring."""

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
    warp_grid_n: int = 5,
) -> dict:
    """One fully-assembled training item: role randomization + context/query
    split + severed-mode flag, on top of one `sample_pair` draw.

    `warp_grid_n`: pass-through to `sample_pair`'s own override of the same
    name -- see that function's docstring. Default (5) is unchanged from
    `sample_warp_pair`/`declared_box`'s own default, so nothing changes
    for an existing caller unless it opts in explicitly.

    `force_rho_zero`: bypass the curriculum and always draw at rho=0 --
    CLAUDE.md's `prior=p0_identity` variant, and the input side of the
    ARCHITECTURE.md build-order go/no-go run (encoder + gated cross-attention
    trained at rho=0 only, before the transport head exists).

    `d`, `n_a_range`, `n_b_range`: pass-through to `sample_pair` -- default
    to ARCHITECTURE.md's full ranges (d in {1,2,3,5}, n_b up to 1024). A
    smaller n_b_range (e.g. configs/experiment/arch_verification.yaml's
    (8, 100)) caps the quadratic cost of decoder self-attention over the
    pooled context [A ; B_inA] (ppfn.prior.registration.dataset.build_pooled_context),
    which the full n_b range can push well past this GPU's memory even at a
    small batch size -- see docs/labbook/ for the OOM this was diagnosed
    from. Fixing `d` (e.g. d=1) also makes the prior/model 1D-plottable per
    .claude/rules/research-demos.md's convention."""
    rho = 0.0 if force_rho_zero else sample_rho_curriculum(rng, progress)
    pair = sample_pair(
        rng, rho=rho, s_max=s_max, d=d, n_a_range=n_a_range, n_b_range=n_b_range,
        warp_grid_n=warp_grid_n,
    )

    role_swapped = bool(rng.random() < 0.35)
    if not role_swapped:
        enc_x, enc_y = pair.x_e_norm, pair.y_e
        dec_x_full, dec_y_full = pair.x_a_norm, pair.y_a
        transport_full = pair.a_inb_target
        # The encoder cloud's OWN points, expressed in the decoder's frame
        # via the true (ground-truth) transport -- the oracle "B_inA"
        # pooling target LUPI-style baselines need at any rho (not just
        # rho=0, unlike ppfn.prior.registration.dataset.build_pooled_context's
        # naive concatenation, which is only correct at rho=0 -- see that
        # function's own docstring). Note this is the OPPOSITE of
        # `transport_full` above: that supervises the decoder cloud's own
        # points mapped INTO the encoder's frame (for the transport head);
        # this is the encoder cloud's points mapped INTO the decoder's
        # frame (for oracle pooling). Both targets already come out of
        # `sample_pair` regardless of role_swapped -- just picking the
        # other one here.
        enc_x_oracle = pair.b_ina_target
    else:
        enc_x, enc_y = pair.x_a_norm, pair.y_a
        dec_x_full, dec_y_full = pair.x_e_norm, pair.y_e
        transport_full = pair.b_ina_target
        enc_x_oracle = pair.a_inb_target

    n_dec = dec_x_full.shape[0]
    k_ctx = int(rng.integers(1, max(2, n_dec)))  # at least 1 context, at least 1 query
    k_ctx = min(k_ctx, n_dec - 1) if n_dec > 1 else 1
    perm = rng.permutation(n_dec)
    ctx_idx, qry_idx = perm[:k_ctx], perm[k_ctx:]

    severed = bool(rng.random() < 0.15)

    return {
        "d_real": pair.d,
        "rho": np.float32(rho),
        "role_swapped": role_swapped,
        "severed": severed,
        "enc_x": enc_x.astype(np.float32),
        "enc_y": enc_y.astype(np.float32),
        "enc_x_oracle": enc_x_oracle.astype(np.float32),
        "dec_ctx_x": dec_x_full[ctx_idx].astype(np.float32),
        "dec_ctx_y": dec_y_full[ctx_idx].astype(np.float32),
        "dec_qry_x": dec_x_full[qry_idx].astype(np.float32),
        "y_qry": dec_y_full[qry_idx].astype(np.float32),
        "transport_ctx": transport_full[ctx_idx].astype(np.float32),
        "transport_qry": transport_full[qry_idx].astype(np.float32),
        "region_type": pair.meta["region_type"],
        "volume_fraction": pair.meta["volume_fraction"],
        "severity_a": pair.meta["severity_a"],
        "severity_b": pair.meta["severity_b"],
        "y_range_fraction": pair.meta["y_range_fraction"],
    }


class RegistrationStreamDataset(IterableDataset):
    """Infinite stream of `build_training_item` draws. One RNG per worker
    (seeded from `seed` + worker id), so different workers don't replay the
    same draws -- standard `IterableDataset` + `get_worker_info` idiom."""

    def __init__(
        self,
        seed: int = 0,
        s_max: float = 0.1,
        force_rho_zero: bool = False,
        progress: SharedProgress | None = None,
        d: int | None = None,
        n_a_range: tuple[int, int] = (8, 256),
        n_b_range: tuple[int, int] = (256, 1024),
        warp_grid_n: int = 5,
    ):
        super().__init__()
        self.seed = seed
        self.s_max = s_max
        self.force_rho_zero = force_rho_zero
        self.progress = progress or SharedProgress(0.0)
        self.d = d
        self.n_a_range = n_a_range
        self.n_b_range = n_b_range
        self.warp_grid_n = warp_grid_n

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
                warp_grid_n=self.warp_grid_n,
                n_a_range=self.n_a_range,
                n_b_range=self.n_b_range,
            )


def _pad_rescale_coords(x: np.ndarray, d_real: int, d_max: int = D_MAX) -> np.ndarray:
    """x: [N, d_real] -> [N, d_max], rescaled by d_max/d_real on the real
    columns then zero-padded -- same fan-in-correction trick as the (dead,
    unrelated-project) `ppfn.model.pfn.pfn._pad_and_rescale_features`, ported
    here for a fixed-width `Linear` layer whose input rows may have fewer
    than `d_max` real coordinates."""
    n = x.shape[0]
    scaled = x * (d_max / d_real)
    if d_real == d_max:
        return scaled
    return np.concatenate(
        [scaled, np.zeros((n, d_max - d_real), dtype=x.dtype)], axis=-1
    )


@dataclass
class RegistrationBatch:
    """Padded, batched tensors -- collate_fn's output. All point-count axes
    (`n_enc`, `n_ctx`, `n_qry`) are padded to the batch max with zeros;
    `*_mask` tensors are True for real tokens, False for padding (the
    `key_padding_mask` convention already used by the existing, unrelated
    `ppfn.model.pfn.pfn.PFN`)."""

    enc_x: torch.Tensor  # [B, n_enc, D_MAX]
    enc_y: torch.Tensor  # [B, n_enc]
    enc_mask: torch.Tensor  # [B, n_enc] bool
    enc_x_oracle: torch.Tensor  # [B, n_enc, D_MAX] -- enc_x pushed into the decoder's frame via the TRUE transport, valid at any rho; same points/mask as enc_x, just different coordinates
    dec_ctx_x: torch.Tensor  # [B, n_ctx, D_MAX]
    dec_ctx_y: torch.Tensor  # [B, n_ctx]
    dec_ctx_mask: torch.Tensor  # [B, n_ctx] bool
    dec_qry_x: torch.Tensor  # [B, n_qry, D_MAX]
    dec_qry_mask: torch.Tensor  # [B, n_qry] bool
    transport_ctx: torch.Tensor  # [B, n_ctx, D_MAX]
    transport_qry: torch.Tensor  # [B, n_qry, D_MAX]
    y_qry: torch.Tensor  # [B, n_qry]
    d_real: torch.Tensor  # [B] long
    rho: torch.Tensor  # [B] float
    role_swapped: torch.Tensor  # [B] bool
    severed: torch.Tensor  # [B] bool
    meta: list  # list[dict], length B, un-batched (region_type etc. for monitors)


def _pad_stack(arrays: list[np.ndarray], max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """arrays: list of [n_i, F] (F fixed across the list). Returns
    (stacked [B, max_n, F], mask [B, max_n] bool, True=real)."""
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


def collate_registration_batch(items: list[dict]) -> RegistrationBatch:
    padded = [
        {
            **it,
            "enc_x": _pad_rescale_coords(it["enc_x"], it["d_real"]),
            "enc_x_oracle": _pad_rescale_coords(it["enc_x_oracle"], it["d_real"]),
            "dec_ctx_x": _pad_rescale_coords(it["dec_ctx_x"], it["d_real"]),
            "dec_qry_x": _pad_rescale_coords(it["dec_qry_x"], it["d_real"]),
            "transport_ctx": _pad_rescale_coords(it["transport_ctx"], it["d_real"]),
            "transport_qry": _pad_rescale_coords(it["transport_qry"], it["d_real"]),
        }
        for it in items
    ]

    max_n_enc = max(it["enc_x"].shape[0] for it in padded)
    max_n_ctx = max(it["dec_ctx_x"].shape[0] for it in padded)
    max_n_qry = max(it["dec_qry_x"].shape[0] for it in padded)

    enc_x, enc_mask = _pad_stack([it["enc_x"] for it in padded], max_n_enc)
    enc_x_oracle, _ = _pad_stack([it["enc_x_oracle"] for it in padded], max_n_enc)
    enc_y, _ = _pad_stack([it["enc_y"][:, None] for it in padded], max_n_enc)
    dec_ctx_x, dec_ctx_mask = _pad_stack([it["dec_ctx_x"] for it in padded], max_n_ctx)
    dec_ctx_y, _ = _pad_stack([it["dec_ctx_y"][:, None] for it in padded], max_n_ctx)
    dec_qry_x, dec_qry_mask = _pad_stack([it["dec_qry_x"] for it in padded], max_n_qry)
    transport_ctx, _ = _pad_stack([it["transport_ctx"] for it in padded], max_n_ctx)
    transport_qry, _ = _pad_stack([it["transport_qry"] for it in padded], max_n_qry)
    y_qry, _ = _pad_stack([it["y_qry"][:, None] for it in padded], max_n_qry)

    return RegistrationBatch(
        enc_x=torch.from_numpy(enc_x),
        enc_y=torch.from_numpy(enc_y).squeeze(-1),
        enc_mask=torch.from_numpy(enc_mask),
        enc_x_oracle=torch.from_numpy(enc_x_oracle),
        dec_ctx_x=torch.from_numpy(dec_ctx_x),
        dec_ctx_y=torch.from_numpy(dec_ctx_y).squeeze(-1),
        dec_ctx_mask=torch.from_numpy(dec_ctx_mask),
        dec_qry_x=torch.from_numpy(dec_qry_x),
        dec_qry_mask=torch.from_numpy(dec_qry_mask),
        transport_ctx=torch.from_numpy(transport_ctx),
        transport_qry=torch.from_numpy(transport_qry),
        y_qry=torch.from_numpy(y_qry).squeeze(-1),
        d_real=torch.tensor([it["d_real"] for it in items], dtype=torch.long),
        rho=torch.tensor([it["rho"] for it in items], dtype=torch.float32),
        role_swapped=torch.tensor(
            [it["role_swapped"] for it in items], dtype=torch.bool
        ),
        severed=torch.tensor([it["severed"] for it in items], dtype=torch.bool),
        meta=[
            {
                "region_type": it["region_type"],
                "volume_fraction": it["volume_fraction"],
                "severity_a": it["severity_a"],
                "severity_b": it["severity_b"],
                "y_range_fraction": it["y_range_fraction"],
            }
            for it in items
        ],
    )


def build_pooled_context(batch: RegistrationBatch) -> RegistrationBatch:
    """`batch` with the decoder context replaced by the pooled context
    `[A ; B_inA]` (ARCHITECTURE.md upper-2, the pooled oracle) -- shared by
    `RegistrationLoss`'s L_pathway, `ArchVerificationLoss`, and
    `ppfn.monitor.arch_verification`, so the three don't drift on this
    construction independently.

    `transport_ctx` is padded with zeros to the pooled shape rather than
    left at its original (smaller) shape: the model always runs its
    (teacher-forced) transport head on every context token regardless of
    `severed_mask`, so this needs SOME value of the right shape. Its content
    is irrelevant to any caller here -- each caller's own severed_mask=True
    forward pass zeroes every cross-attention gate, so this pooled batch is
    only ever used to read off `predictive_logits`, never a transport
    target."""
    pooled_x = torch.cat([batch.dec_ctx_x, batch.enc_x], dim=1)
    pooled_y = torch.cat([batch.dec_ctx_y, batch.enc_y], dim=1)
    pooled_mask = torch.cat([batch.dec_ctx_mask, batch.enc_mask], dim=1)
    pooled_transport = torch.cat(
        [batch.transport_ctx, torch.zeros_like(batch.enc_x)], dim=1
    )
    return replace(
        batch,
        dec_ctx_x=pooled_x,
        dec_ctx_y=pooled_y,
        dec_ctx_mask=pooled_mask,
        transport_ctx=pooled_transport,
    )


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build one batch and
    print every tensor's shape/dtype, so padding/masking is eyeballable."""
    from torch.utils.data import DataLoader

    dataset = RegistrationStreamDataset(seed=0, s_max=0.1)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_registration_batch)
    batch = next(iter(loader))
    for name in (
        "enc_x",
        "enc_y",
        "enc_mask",
        "dec_ctx_x",
        "dec_ctx_y",
        "dec_ctx_mask",
        "dec_qry_x",
        "dec_qry_mask",
        "transport_ctx",
        "transport_qry",
        "y_qry",
        "d_real",
        "rho",
        "role_swapped",
        "severed",
    ):
        t = getattr(batch, name)
        print(f"{name:16s} {tuple(t.shape)} {t.dtype}")
    print("meta[0]:", batch.meta[0])
