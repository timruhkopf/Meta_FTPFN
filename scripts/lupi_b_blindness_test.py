"""B-blindness ablation: does LUPIPFN's student mode actually use B's
specific content, or would ANY B do? For N draws, compute student-mode NLL
on A's queries with (a) the pair's own, matching B and (b) an unrelated
B swapped in (same n_b range, completely different draw/seed) -- if NLL
barely changes, the model isn't reading B's content."""
import dataclasses

import numpy as np
import torch

from ppfn.model.lupi.model import LUPIPFN
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch, collate_lupi_batch
from ppfn.prior.lupi.sampler import sample_pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lupi = LUPIPFN(d_model=256, n_heads=8, n_layers_enc_b=3, n_layers_align=6,
               d_ff=512, n_bins_predictive=64, dropout=0.0).to(device)
ckpt = torch.load(
    "/home/ruhkopf/PycharmProjects/Meta_FTPFN_lupi/models/01-pretraining-lupi-baseline/best_01-pretraining-lupi-baseline.pt",
    map_location=device, weights_only=False,
)
lupi.load_state_dict(ckpt["model_state_dict"])
lupi.eval()
print("checkpoint epoch:", ckpt.get("epoch"), "best_score:", ckpt.get("best_score"))


def item_from_pair(pair):
    return {
        "d_real": pair.d, "rho": np.float32(pair.rho), "beta": np.float32(pair.beta),
        "enc_x": pair.x_b.astype(np.float32), "enc_z": pair.z_b.astype(np.float32),
        "enc_x_inA": pair.x_b_inA.astype(np.float32),
        "dec_ctx_x": pair.x_a_ctx.astype(np.float32), "dec_ctx_z": pair.z_a_ctx.astype(np.float32),
        "dec_ctx_oracle_bpos": pair.oracle_bpos_a_ctx.astype(np.float32),
        "dec_qry_x": pair.x_a_qry.astype(np.float32), "dec_qry_z": pair.z_a_qry.astype(np.float32),
        "dec_qry_oracle_bpos": pair.oracle_bpos_a_qry.astype(np.float32),
        "dec_qry_source": pair.qry_source,
        "region_type": pair.meta["region_type"], "volume_fraction": pair.meta["volume_fraction"],
    }


def move(batch, device):
    return LUPIBatch(**{
        f.name: (getattr(batch, f.name).to(device) if isinstance(getattr(batch, f.name), torch.Tensor) else getattr(batch, f.name))
        for f in dataclasses.fields(batch)
    })


N = 60
SEED0 = 2000
nll_real, nll_swapped, nll_oracle = [], [], []

rng_target = np.random.default_rng(SEED0)
rng_foreign = np.random.default_rng(SEED0 + 1_000_000)

for i in range(N):
    d = int(rng_target.choice([1, 2, 3, 5]))
    rho = float(rng_target.uniform(0.0, 1.0))
    pair_t = sample_pair(rng_target, rho=rho, d=d, s_max=0.1,
                          n_a_range=(8, 100), n_b_range=(8, 100), n_qry_range=(8, 64))
    # unrelated foreign pair -- independent draw, own d/rho, only used for its B cloud
    d_f = int(rng_foreign.choice([1, 2, 3, 5]))
    rho_f = float(rng_foreign.uniform(0.0, 1.0))
    pair_f = sample_pair(rng_foreign, rho=rho_f, d=d_f, s_max=0.1,
                          n_a_range=(8, 100), n_b_range=(8, 100), n_qry_range=(8, 64))

    item_real = item_from_pair(pair_t)
    batch_real = move(collate_lupi_batch([item_real]), device)

    # swap in the foreign B (own x/z/padding -- independent d_real, that's fine,
    # collate pads everything to D_MAX regardless).
    item_swapped = dict(item_real)
    item_swapped["enc_x"] = pair_f.x_b.astype(np.float32)
    item_swapped["enc_z"] = pair_f.z_b.astype(np.float32)
    item_swapped["enc_x_inA"] = pair_f.x_b_inA.astype(np.float32)  # unused by student/oracle forward, kept for schema
    batch_swapped = move(collate_lupi_batch([item_swapped]), device)

    with torch.no_grad():
        b_real = lupi.encode_b(batch_real)
        out_real = lupi.align(batch_real, b_real, mode="student")
        out_oracle = lupi.align(batch_real, b_real, mode="oracle")

        b_swap = lupi.encode_b(batch_swapped)
        out_swap = lupi.align(batch_swapped, b_swap, mode="student")

    mask = batch_real.dec_qry_mask.float()
    denom = mask.sum().clamp_min(1.0)

    nll_r = (lupi.bar_dist(out_real["predictive_logits"], batch_real.dec_qry_z) * mask).sum() / denom
    nll_s = (lupi.bar_dist(out_swap["predictive_logits"], batch_real.dec_qry_z) * mask).sum() / denom
    nll_o = (lupi.bar_dist(out_oracle["predictive_logits"], batch_real.dec_qry_z) * mask).sum() / denom

    nll_real.append(nll_r.item())
    nll_swapped.append(nll_s.item())
    nll_oracle.append(nll_o.item())

nll_real = np.array(nll_real)
nll_swapped = np.array(nll_swapped)
nll_oracle = np.array(nll_oracle)

print(f"N={N} draws")
print(f"student, real B    : mean NLL = {nll_real.mean():.4f}  (std {nll_real.std():.4f})")
print(f"student, swapped B : mean NLL = {nll_swapped.mean():.4f}  (std {nll_swapped.std():.4f})")
print(f"oracle,  real B     : mean NLL = {nll_oracle.mean():.4f}  (std {nll_oracle.std():.4f})")
print()
print(f"delta (swapped - real), student: mean {np.mean(nll_swapped - nll_real):.4f}  paired std {np.std(nll_swapped - nll_real):.4f}")
print(f"delta (real - oracle), student vs oracle: mean {np.mean(nll_real - nll_oracle):.4f}")
from scipy import stats
t, p = stats.ttest_rel(nll_swapped, nll_real)
print(f"paired t-test (swapped vs real), t={t:.3f}, p={p:.2e}")
