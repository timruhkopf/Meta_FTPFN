"""
Core idea is that OT would fail us due to conservation of mass.
In the translation task of Vaswani, we know that the meaning of the text should be identical
which also admits the conservation of mass property.
This would admit a prompt like OT as in (How do transformers align tokens? -- Provable Optimal Transport with Transformers
-- Hadi Daneshmand) and some amortized transport operator like (In-Context Operator Learning on the Space of Probability Measures Frank Cole)
But ultimately, they are limited by 1to1 mappings; i.e. both sequences must be of the same size
and the conservation does not allow us to use new information from the more dense task.


However, in our case, we basically have a language translation task, where one sentence is informationally longer than the
other; which is basically the same thing as the original autoregressive decoding did; it prepared the self attn on a shorter
sequence to read off of a longer sequence with more content; there it basically needs to assess, what information is already
contained in on itself (partial english sentence), and what information is still missing from the related context
(complete French sentence). We basically want the same thing, but with a PFN backend.

The decoder is classic PFN; i.e. it has row attn (and column attn + thinking rows?); specifically
* train-train self attn (A_train)
* test-train cross attn (A_test)

Then, we can pass B through the encoder stack of the same PFN backend (different instance), but ignore the test-train
cross attn here (by not passing B_test and by setting SEP=None) Then we basically only need to implement cross attn
from A_test to B and A_train to B, and finally have A_test predict bar logits (same as PFN).

A few ideas:
* since both A and B basically are both valid functions in the sense of the prior, at least initially, the train-self
attn could be shared between encoder and decoder to stabalize learning.
* feature attn might contribute; **cross feature attn** could also help
* we could try and allow B to cross attend to A
* we could try using OT prompt + special attn weights on the thinking tokens of both domains.
* during training, we know what B_inA is exactly, so we can concatenate that to A_Test, penalizing the model where it matters.

Note that Vaswani had the decoder cross attend at the different layers to the final output (not the interpediate layer outputs)
of the encoder.
"""


"""
TabPFN-native Encoder-Decoder Transfer Model
=============================================

Rebuild of the dual-stream transfer architecture directly on top of TabPFN
v2.5's real building blocks, per the latest revision:

  - `TabPFNBlock` (row/feature attention + column/cell attention + MLP) is
    imported and used AS-IS, not reimplemented. It is the single reusable
    layer type for both streams.
  - Cell encoding produces a (B, R, C, E) tensor -- batch, rows, columns
    (feature groups + 1 target column), embedding -- matching exactly what
    `TabPFNBlock` expects, so no shape-adaptation glue is needed between the
    encoder used here and the imported blocks.
  - DECODER STACK (stream A): `TabPFNBlock` used exactly as in TabPFN v2.5,
    with a real `single_eval_pos`. This gives A_train <-> A_train self
    attention and A_test -> A_train cross attention for free, via the
    block's built-in column-attention masking -- no separate code needed for
    that part.
  - ENCODER STACK (stream B): the SAME `TabPFNBlock`, but called with
    `single_eval_pos=None`, which bypasses the train/test masking entirely
    (full bidirectional attention over all of B). This is the "no context
    split" mode -- B is just encoded, not PFN-queried.
  - VASWANI CROSS-ATTENTION: a genuinely new sublayer (TabPFN has no
    multi-stream cross-attention, so this part cannot be imported) inserted
    after each decoder `TabPFNBlock`. It applies the *same* cross-attention,
    with the *same* weights, to every decoder row -- A_train rows and A_test
    rows alike -- reading from the (single, precomputed) encoder memory of
    B. This mirrors vanilla Vaswani: the encoder runs once, and every
    decoder layer attends into that same fixed memory.
  - OUTPUT: a linear head to `num_bars` logits, scored with TabPFN's own
    `FullSupportBarDistribution` NLL (imported, not reimplemented).
  - B_in_A test points are not a separate loss term with their own decode
    pass -- they are literally concatenated onto A's test-query rows before
    the single decoder forward, so they're PFN-conditioned on A_train
    exactly like any other A_test row, cross-attend into B exactly like any
    other A_test row, and only differ in which ground truth (`Y_test_A` vs
    `Y_test_B_in_A`) is used to score them.

Both the fused decode (with cross-attention into B) and a marginal decode
(same decoder weights, cross-attention sublayer skipped) are computed, so
the "what would an A-only PFN have said" comparison from before still falls
out directly -- now on real TabPFN blocks rather than hand-rolled attention.
"""

# from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from tabpfn.architectures.tabpfn_v2_5 import (
    TabPFNBlock,
    AddThinkingRows,
    NAN_INDICATOR,
)
from tabpfn.architectures.shared.bar_distribution import (
    FullSupportBarDistribution,
    get_bucket_limits,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    x_dim: int = 1
    emsize: int = 128
    nhead: int = 4
    dim_feedforward_mult: int = 2
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    num_thinking_rows_B: int = 16  # (a.1)/(c): extra tokens, not a bottleneck -- raw B rows remain
    num_thinking_rows_A: int = 8
    num_bars: int = 128            # bar-distribution resolution for the NLL head
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Cell encoding: (x, y, has_label) -> (B, R, C, E)
# ---------------------------------------------------------------------------
class CellEncoder(nn.Module):
    """Embeds x (x_dim columns) and y (1 target column) into TabPFN's cell
    grid layout. Uses the same [value, nan_indicator] encoding trick as the
    real v2.5 encoder (`NAN_INDICATOR` imported directly) so query/test rows
    carry an explicit "missing" flag on the target column rather than a
    silent zero.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.feature_embedder = nn.Linear(2, cfg.emsize, bias=False)  # shared across x columns
        self.target_embedder = nn.Linear(2, cfg.emsize, bias=False)

    def forward(self, x_TBX: torch.Tensor, y_TB1: torch.Tensor | None, has_label_TB1: torch.Tensor) -> torch.Tensor:
        T, B, X = x_TBX.shape
        device = x_TBX.device

        x_feat = torch.stack([x_TBX, torch.zeros_like(x_TBX)], dim=-1)          # [T,B,X,2] -- x always observed
        x_cols_TBXE = self.feature_embedder(x_feat)                            # [T,B,X,E]

        if y_TB1 is None:
            y_TB1 = torch.zeros(T, B, 1, device=device)
        has_label = has_label_TB1.bool()
        y_masked = torch.where(has_label, y_TB1, torch.zeros_like(y_TB1))
        nan_ind = torch.where(has_label, torch.zeros_like(y_TB1), torch.full_like(y_TB1, NAN_INDICATOR))
        y_feat = torch.stack([y_masked, nan_ind], dim=-1)                      # [T,B,1,2]
        y_col_TB1E = self.target_embedder(y_feat)                              # [T,B,1,E]

        cell_TBCE = torch.cat([x_cols_TBXE, y_col_TB1E], dim=2)                # [T,B,C=X+1,E]
        return cell_TBCE.permute(1, 0, 2, 3).contiguous()                      # -> [B,R,C,E]


# ---------------------------------------------------------------------------
# Vaswani-style cross-attention sublayer (not in TabPFN -- there is no
# multi-stream primitive to import for this part)
# ---------------------------------------------------------------------------
class CrossAttnSublayer(nn.Module):
    # FIXME: verify, that tabpfn uses this cell encoding!
    """Applied uniformly to every decoder row (train and test alike) -- no
    masking distinction, satisfying 'the same exact cross-attention scheme
    from A_train and A_test to B'. Reads from a single, fixed encoder
    memory, as in vanilla Vaswani (the encoder runs once; every decoder
    layer attends into that same output).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim_ff = cfg.emsize * cfg.dim_feedforward_mult
        self.ln_q = nn.LayerNorm(cfg.emsize, elementwise_affine=False)
        self.ln_kv = nn.LayerNorm(cfg.emsize, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(cfg.emsize, cfg.nhead, dropout=cfg.dropout, batch_first=True)
        # FIXME: what if we used the TABPFNBlock once more -- but for cross attending? -- allowing feature cross attn?
        self.ln_ff = nn.LayerNorm(cfg.emsize, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(cfg.emsize, dim_ff, bias=False), nn.GELU(), nn.Linear(dim_ff, cfg.emsize, bias=False)
        )
        # zero-init output projections -- matches TabPFNBlock's convention so
        # every new sublayer starts close to identity for stable training
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.ff[-1].weight)

    def forward(self, q_BRE: torch.Tensor, memory_BSE: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(q_BRE)
        kv = self.ln_kv(memory_BSE)
        # FIXME: are we attn Atest to [A, B] ? this way using softmax, the model can choose to ignore B
        # TODO consider adding task tokens?
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        q_BRE = q_BRE + attn_out
        q_BRE = q_BRE + self.ff(self.ln_ff(q_BRE))
        return q_BRE


# ---------------------------------------------------------------------------
# Encoder / Decoder stacks
# ---------------------------------------------------------------------------
class EncoderStack(nn.Module):
    """
    Proesses B; the related task using Tabpfnblocks with thinking rows, is preparing it for cross attention
    complete 'french' sentence.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim_ff = cfg.emsize * cfg.dim_feedforward_mult
        self.add_thinking_rows = AddThinkingRows(cfg.num_thinking_rows_B, cfg.emsize)
        self.blocks = nn.ModuleList(
            # FIXME: should the first TabPFNBlocks of encoder and decoder be coupled; i.e. weight share; since both
            #  need to understand their own geometry first anyways?.
            TabPFNBlock(emsize=cfg.emsize, nhead=cfg.nhead, dim_feedforward=dim_ff)
            for _ in range(cfg.n_encoder_layers)
        )

    def forward(self, cell_BRCE: torch.Tensor) -> torch.Tensor:
        x, _ = self.add_thinking_rows(cell_BRCE, single_eval_pos=cell_BRCE.shape[1])
        # TODO What if B were to featurewise cross attend to A?
        for block in self.blocks:
            # Wrap x in a list *right here* on every iteration
            x, _ = block([x], single_eval_pos=None, save_peak_memory_factor=None)
        return x  # returns the tensor directly


class DecoderStack(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dim_ff = cfg.emsize * cfg.dim_feedforward_mult
        self.add_thinking_rows = AddThinkingRows(cfg.num_thinking_rows_A, cfg.emsize) # Fixme: Add thinking rows here? why?
        self.self_blocks = nn.ModuleList(
            TabPFNBlock(emsize=cfg.emsize, nhead=cfg.nhead, dim_feedforward=dim_ff)
            for _ in range(cfg.n_decoder_layers)
        )
        self.cross_blocks = nn.ModuleList(
            CrossAttnSublayer(cfg) for _ in range(cfg.n_decoder_layers)
        )

    def forward(
            self, cell_BRCE: torch.Tensor, n_train: int, memory_row_BSE: torch.Tensor | None, *,
            run_marginal: bool = False
    ) -> tuple[torch.Tensor, int]:
        # TODO: consider: what if we allowed the decoder to look at the intermediate outputs of the encoder stack instead?
        x, single_eval_pos = self.add_thinking_rows(cell_BRCE, single_eval_pos=n_train) # FIXME: add thinking rows here? why?

        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            # Wrap x in a list inline
            x, _ = self_block([x], single_eval_pos=single_eval_pos, save_peak_memory_factor=None)

            if not run_marginal: # bypassing the cross block, if we only want to run marginally (decoder only!)!
                assert memory_row_BSE is not None
                row_repr_BRE = x[:, :, -1, :]  # target column = per-row summary
                row_repr_BRE = cross_block(row_repr_BRE, memory_row_BSE)
                x = torch.cat([x[:, :, :-1, :], row_repr_BRE.unsqueeze(2)], dim=2)

        return x, single_eval_pos
# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------
class TabPFNTransferModel(nn.Module):
    def __init__(self, cfg: ModelConfig, y_borders: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.embed_A = CellEncoder(cfg)
        self.embed_B = CellEncoder(cfg)  # staged: kept separate for now, see TODO below
        # TODO(share-weights): tie embed_A/embed_B and the first decoder/encoder
        # block once basic sharing is worth staging in -- both streams are
        # draws from the same f_theta prior, just windowed differently.
        self.encoder = EncoderStack(cfg)
        self.decoder = DecoderStack(cfg)

        dim_ff = cfg.emsize * cfg.dim_feedforward_mult
        self.output_head = nn.Sequential(
            nn.Linear(cfg.emsize, dim_ff), nn.GELU(), nn.Linear(dim_ff, cfg.num_bars)
        )
        self.bar_dist = FullSupportBarDistribution(y_borders)

    def encode_B(self, X_B: torch.Tensor, Y_B: torch.Tensor) -> torch.Tensor:
        T, B, _ = X_B.shape
        has_label = torch.ones(T, B, 1, device=X_B.device)
        cell_BRCE = self.embed_B(X_B, Y_B, has_label)
        enc_out_BRCE = self.encoder(cell_BRCE)
        return enc_out_BRCE[:, :, -1, :]  # [B, R_B, E] row memory read from target column

    def build_A_query_cells(self, X_A_train, Y_A_train, X_query, batch_size, device):
        n_train = X_A_train.shape[0]
        n_query = X_query.shape[0]
        has_label_train = torch.ones(n_train, batch_size, 1, device=device)
        has_label_query = torch.zeros(n_query, batch_size, 1, device=device)

        train_cell = self.embed_A(X_A_train, Y_A_train, has_label_train)
        query_cell = self.embed_A(X_query, None, has_label_query)
        return torch.cat([train_cell, query_cell], dim=1), n_train  # concat along row dim

    def forward(self, batch: dict) -> dict:
        """`batch` has the exact `InfiniteHarmonicsStream._sample_batch` shape.
        Tensors are assumed to already be on the target device."""
        # FIXME: pass in explicitly and parse the batch in the train loop
        tr, te = batch["train"], batch["test"]
        device = tr["X_A"].device
        batch_size = tr["X_A"].shape[1]

        memory_B = self.encode_B(tr["X_B_obs"], tr["Y_B_obs"])  # [B, R_B, E]

        # B_in_A test points are literally appended to A's query rows here --
        # a single combined query batch, one decoder forward.
        X_query = torch.cat([te["X_A"], tr["X_B_in_A"]], dim=0)
        n_test_A = te["X_A"].shape[0]

        cell_A, n_train = self.build_A_query_cells(tr["X_A"], tr["Y_A"], X_query, batch_size, device)

        fused_out, sep = self.decoder(cell_A, n_train, memory_B, run_marginal=False)
        # FIXME: rather than passing the baseline model through here explicitly, we could just not pass X_B, sidestepping
        #  the encoder entirely and do that in the same batch? -- this way, we ensure, that the decoder only model will
        #  do valid inference and is the actual on-board baseline!  We can of course also ask for different contexts;
        #  i.e. do a pass of B vs B_test but through the decoder only (ensuring we will generalize the decoder to this
        #  context length!) Then we can also use the model in decoder only as a validation callback as well!
        marg_out, _ = self.decoder(cell_A, n_train, None, run_marginal=True)

        # fused_out / marg_out are [B, R, C, E]; test rows sit along dim=1 from `sep` on.
        fused_test_repr = fused_out[:, sep:, -1, :]    # [B, n_test_total, E]
        marg_test_repr = marg_out[:, sep:, -1, :]

        fused_logits = self.output_head(fused_test_repr).transpose(0, 1)  # [n_test_total, B, num_bars]
        marg_logits = self.output_head(marg_test_repr).transpose(0, 1)

        return { # FIXME: this split won't make sense during deployment!
            "fused_logits_A": fused_logits[:n_test_A],
            "fused_logits_BinA": fused_logits[n_test_A:],
            "marg_logits_A": marg_logits[:n_test_A],
            "marg_logits_BinA": marg_logits[n_test_A:],
            # FIXME: do these other metrics make sense? -- while they indicate whether or not we are reconstrucing y correctly
            #  based on the split -- i.e. A_test or B_inA as test points, it is only really meaningful to compare against the gt
            #  marginal PFN; i.e. using only A to predict A_test and B_inA test.
            #  Can we somehow in parallel train a regular marginal PFN, so that the baseline is baked into the model?
            #  no cross attn is allowed? is the same as not passing B and still doing a fwd pass on the decoder only,
            #  then we can compare against
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_loss(model: TabPFNTransferModel, out: dict, batch: dict, lambda_marginal: float = 0.3):
    te = batch["test"]
    tr = batch["train"]
    Y_A = te["Y_A"].squeeze(-1)
    Y_BinA = tr["Y_B_in_A"].squeeze(-1)

    l_fused_A = model.bar_dist(out["fused_logits_A"], Y_A)          # [n_test_A, B]
    l_fused_BinA = model.bar_dist(out["fused_logits_BinA"], Y_BinA)  # the focused, penalized target
    l_marg_A = model.bar_dist(out["marg_logits_A"], Y_A)
    l_marg_BinA_only = model.bar_dist(out["marg_logits_BinA"], Y_BinA)  # A-only baseline at B_in_A locations

    loss = l_fused_BinA.mean() + lambda_marginal * (l_fused_A.mean() + l_marg_A.mean())

    with torch.no_grad():
        # positive => cross-attending into B genuinely beat an A-only PFN
        # at the exact same query locations
        gap_per_point = l_marg_BinA_only - l_fused_BinA

    logs = {
        "loss": loss.item(),
        "nll_fused_A": l_fused_A.mean().item(),
        "nll_fused_BinA": l_fused_BinA.mean().item(),
        "nll_marg_A": l_marg_A.mean().item(),
        "nll_marg_BinA_only": l_marg_BinA_only.mean().item(),
        "fused_vs_marginal_gap": gap_per_point.mean().item(),
    }
    return loss, logs, gap_per_point


def stratify_by_relatedness(gap_per_point: torch.Tensor, is_unrelated: torch.Tensor) -> dict:
    is_unrelated = is_unrelated.to(gap_per_point.device).bool()
    batch_gap = gap_per_point.mean(dim=0)  # mean over test rows, keep batch dim
    related = batch_gap[~is_unrelated].mean().item() if (~is_unrelated).any() else float("nan")
    unrelated = batch_gap[is_unrelated].mean().item() if is_unrelated.any() else float("nan")
    return {"gap_related": related, "gap_unrelated": unrelated}


# ---------------------------------------------------------------------------
# Border construction (from imported get_bucket_limits)
# ---------------------------------------------------------------------------
def build_y_borders(num_bars: int, y_samples: torch.Tensor | None = None, y_range: tuple[float, float] | None = None) -> torch.Tensor:
    """Pass either a sample of y values from the prior (preferred -- gives
    equal-mass bars) or a fixed (min, max) range."""
    if y_samples is not None:
        return get_bucket_limits(num_outputs=num_bars, ys=y_samples)
    assert y_range is not None
    return get_bucket_limits(num_outputs=num_bars, full_range=y_range)


# ---------------------------------------------------------------------------
# Minimal training loop skeleton
# ---------------------------------------------------------------------------
def move_batch_to_device(batch, device):
    batch["train"] = {k: v.to(device) for k, v in batch["train"].items()}
    batch["test"] = {k: v.to(device) for k, v in batch["test"].items()}
    return batch


def train(model, data_stream, steps=10_000, lr=3e-4, device="cuda"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    it = iter(data_stream)
    for step in range(steps):
        batch = move_batch_to_device(next(it), device)
        out = model(batch)
        loss, logs, gap = compute_loss(model, out, batch)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            strat = stratify_by_relatedness(gap, batch["params"]["is_unrelated"])
            print(step, {**logs, **strat})


if __name__ == "__main__":
    from ppfn.prior.harmonics_fix.harmonic_mixture_prior import HarmonicMixturePrior
    from ppfn.prior.harmonics_fix.stream_dataset import InfiniteHarmonicsStream

    cfg = ModelConfig(emsize=32, nhead=4, n_encoder_layers=2, n_decoder_layers=2, num_bars=100)
    borders = build_y_borders(cfg.num_bars, y_range=(-7.0, 7.0))
    model = TabPFNTransferModel(cfg, borders)

    prior = HarmonicMixturePrior()

    dataset = InfiniteHarmonicsStream(prior, batch_size=128, n_test=128)


    train(model, dataset, steps=100000, )
    print(model)

    """
    n_A 20  n_B=50  WSS
    17800 {'loss': 0.7034624814987183, 'nll_fused_A': 0.31111711263656616, 'nll_fused_BinA': 0.31593436002731323, 'nll_marg_A': 0.9806432723999023, 'nll_marg_BinA_only': 0.9793418645858765, 'fused_vs_marginal_gap': 0.6634074449539185, 'gap_related': 0.6634074449539185, 'gap_unrelated': nan}
    99900 {'loss': -0.33295631408691406, 'nll_fused_A': -0.4275573492050171, 'nll_fused_BinA': -0.4238017201423645, 'nll_marg_A': 0.7303754091262817, 'nll_marg_BinA_only': 0.7495419979095459, 'fused_vs_marginal_gap': 1.1733436584472656, 'gap_related': 1.1733437776565552, 'gap_unrelated': nan}

    
    batch_size 128 n_A 10 n_B=50  WSS
    56100 {'loss': 0.6290130615234375, 'nll_fused_A': 0.16290147602558136, 'nll_fused_BinA': 0.1602393388748169, 'nll_marg_A': 1.3996775150299072, 'nll_marg_BinA_only': 1.405665636062622, 'fused_vs_marginal_gap': 1.2454262971878052, 'gap_related': 1.2454262971878052, 'gap_unrelated': nan}
    98500 {'loss': -0.28215423226356506, 'nll_fused_A': -0.538619875907898, 'nll_fused_BinA': -0.5358275175094604, 'nll_marg_A': 1.384197473526001, 'nll_marg_BinA_only': 1.3834961652755737, 'fused_vs_marginal_gap': 1.9193236827850342, 'gap_related': 1.9193236827850342, 'gap_unrelated': nan}

    
    # 100000 steps, batch_size 32, n_A = 32, n_B = 64 WSS
    99900 {'loss': -0.6193640232086182, 'nll_fused_A': -0.49316561222076416, 'nll_fused_BinA': -0.48550164699554443, 'nll_marg_A': 0.04695780202746391, 'nll_marg_BinA_only': 0.0504516065120697, 'fused_vs_marginal_gap': 0.5359532833099365, 'gap_related': 0.5359532833099365, 'gap_unrelated': nan}

    
    An Open question remains: 
    what can we use as "model-evidence" here; i.e. when we have multiple tasks B, and pass them through in batch parallel 
    against A, and we therefore get A|B_i, how to we create a BMA posterior over them? 
    At that point, the we will also worry about unrelated tasks, and that the model must be capable of rejecting it 
    i.e. e.g. the cross attn, no longer can be "pure" i.e. 
    
    an interesting avenue is actually looking into the cross-feature attn more indepth, because this is closest to what 
    we'd expect to act on feature space alignment; so if B has access to A's column statistics, it could potentially warp 
    itself directly into it, and be considerably easier for A to read; but this probably means, that we need to actually 
    have B process itself in parallel to A. This is a bit more involved. 
    
    Main claim; meta tasks in HPO have task specific properties, that change the space itself; i.e. warp / shift x
    and can do output transformations as well; i.e. scale / shift warp(y, x). 
    Simple Row attention won't suffice to 
     
    Baselines: 
    # (0) Trained from scratch using only the decoder module (never with the encoder to be a tidy baseline)
    # (1) Check what the decoder only prediction will be; i.e. during training with cross attn, how much can we still 
    #  trust the marginal branch? ; i.e. do we need to train a separate PFN again?
    * PFN(A, A_test)  
    * PFN([A, B_inA], A_test)  --> what happens if we use the encoder(B_inA) and the decoder(A, A_test)? --> if they 
    are from the same domain, should they not be recognized as such and act as if they were indeed compatible?)
    
    # independent Baseline
    * MTPFN: (we can improve it, by optinally allowing it to have feat attn as well)
    
    
    Benchmarks: 
    # (0) Synthetic / Prior training: 
    * harmonics 1d + plotting
    * BNN_theta, with linear interpolation between them?
    
    # (1) Real benchmarks
    * HPOBench (using MTPFN's meta train test split)
    * MFPBench (Using something similar to the meta train test split above) --> will require to train on IFBO prior, with 
    
    
    # REMARK: 
    while interesting, a connection to OT is a bit impractical. 
    however, there is literature on OT being recoverable as prompt engineering with special attn (which is elegant) 
    but ultimately the mass conservation is problematic, because it might erase information from the original space
    """

    import re
    import matplotlib.pyplot as plt
    import pandas as pd


    def plot_learning_curves(
            log_text: str, metrics_to_plot: list = None, title: str = "Learning Curves"
    ):
        """Parses Python console log output and plots learning curves in one plot.

        Parameters:
        - log_text (str): Raw log output from console.
        - metrics_to_plot (list[str] or None): List of metric keys to plot (e.g., ['loss', 'nll_fused_A']).
                                               If None, plots all metrics that contain numeric values.
        - title (str): Plot title.
        """
        records = []

        # Parse line by line: <step> {<dictionary_of_metrics>}
        for line in log_text.strip().split("\n"):
            line = line.strip()
            match = re.match(r"^\s*(\d+)\s+(\{.*\})\s*$", line)
            if match:
                step = int(match.group(1))
                dict_str = match.group(2)

                # Handle Python 'nan' strings safely
                dict_str_clean = re.sub(r"\bnan\b", 'float("nan")', dict_str)

                try:
                    metrics = eval(dict_str_clean)
                    metrics["step"] = step
                    records.append(metrics)
                except Exception:
                    continue

        df = pd.DataFrame(records)
        if df.empty:
            print("No valid log entries found.")
            return

        df = df.set_index("step")

        # Default to all metrics with valid numeric data
        if metrics_to_plot is None:
            metrics_to_plot = [col for col in df.columns if df[col].notna().any()]

        # Plotting
        plt.figure(figsize=(10, 6))

        for metric in metrics_to_plot:
            if metric in df.columns:
                plt.plot(df.index, df[metric], label=metric, linewidth=1.5)
            else:
                print(f"Warning: Metric '{metric}' not found in data.")

        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


    log_output = """
    0 {'loss': 4.223830699920654, 'nll_fused_A': 2.637399196624756, 'nll_fused_BinA': 2.6413910388946533, 'nll_marg_A': 2.637399196624756, 'nll_marg_BinA_only': 2.6413910388946533, 'fused_vs_marginal_gap': 0.0, 'gap_related': 0.0, 'gap_unrelated': nan}
    100 {'loss': 3.358975410461426, 'nll_fused_A': 2.08329439163208, 'nll_fused_BinA': 2.0857431888580322, 'nll_marg_A': 2.160813093185425, 'nll_marg_BinA_only': 2.1626763343811035, 'fused_vs_marginal_gap': 0.07693322002887726, 'gap_related': 0.07693322002887726, 'gap_unrelated': nan}
    200 {'loss': 3.2633743286132812, 'nll_fused_A': 2.0305559635162354, 'nll_fused_BinA': 2.031294107437134, 'nll_marg_A': 2.0763778686523438, 'nll_marg_BinA_only': 2.0761303901672363, 'fused_vs_marginal_gap': 0.044836126267910004, 'gap_related': 0.044836126267910004, 'gap_unrelated': nan}
    300 {'loss': 3.2098374366760254, 'nll_fused_A': 2.002521276473999, 'nll_fused_BinA': 1.9977716207504272, 'nll_marg_A': 2.0376977920532227, 'nll_marg_BinA_only': 2.033859968185425, 'fused_vs_marginal_gap': 0.036088503897190094, 'gap_related': 0.036088503897190094, 'gap_unrelated': nan}
    400 {'loss': 3.202589988708496, 'nll_fused_A': 1.9985644817352295, 'nll_fused_BinA': 1.993132472038269, 'nll_marg_A': 2.0329599380493164, 'nll_marg_BinA_only': 2.0317559242248535, 'fused_vs_marginal_gap': 0.03862346336245537, 'gap_related': 0.03862345963716507, 'gap_unrelated': nan}
    500 {'loss': 3.121015787124634, 'nll_fused_A': 1.9490936994552612, 'nll_fused_BinA': 1.942697525024414, 'nll_marg_A': 1.9786336421966553, 'nll_marg_BinA_only': 1.9755516052246094, 'fused_vs_marginal_gap': 0.03285413980484009, 'gap_related': 0.03285413980484009, 'gap_unrelated': nan}
    600 {'loss': 3.065291404724121, 'nll_fused_A': 1.91536545753479, 'nll_fused_BinA': 1.9039435386657715, 'nll_marg_A': 1.9557934999465942, 'nll_marg_BinA_only': 1.9448647499084473, 'fused_vs_marginal_gap': 0.040921375155448914, 'gap_related': 0.040921375155448914, 'gap_unrelated': nan}
    700 {'loss': 3.1369802951812744, 'nll_fused_A': 1.9522199630737305, 'nll_fused_BinA': 1.9569861888885498, 'nll_marg_A': 1.9810932874679565, 'nll_marg_BinA_only': 1.9832313060760498, 'fused_vs_marginal_gap': 0.02624506503343582, 'gap_related': 0.02624506503343582, 'gap_unrelated': nan}
    800 {'loss': 3.0390801429748535, 'nll_fused_A': 1.8852145671844482, 'nll_fused_BinA': 1.8936612606048584, 'nll_marg_A': 1.9328480958938599, 'nll_marg_BinA_only': 1.9399960041046143, 'fused_vs_marginal_gap': 0.04633477330207825, 'gap_related': 0.046334777027368546, 'gap_unrelated': nan}
    900 {'loss': 2.976773262023926, 'nll_fused_A': 1.8474090099334717, 'nll_fused_BinA': 1.847521185874939, 'nll_marg_A': 1.916764259338379, 'nll_marg_BinA_only': 1.915462613105774, 'fused_vs_marginal_gap': 0.06794146448373795, 'gap_related': 0.06794146448373795, 'gap_unrelated': nan}
    1000 {'loss': 2.9123072624206543, 'nll_fused_A': 1.800004005432129, 'nll_fused_BinA': 1.7989232540130615, 'nll_marg_A': 1.9112753868103027, 'nll_marg_BinA_only': 1.9116374254226685, 'fused_vs_marginal_gap': 0.11271411180496216, 'gap_related': 0.11271411180496216, 'gap_unrelated': nan}
    1100 {'loss': 2.8443708419799805, 'nll_fused_A': 1.7398831844329834, 'nll_fused_BinA': 1.7465202808380127, 'nll_marg_A': 1.919618844985962, 'nll_marg_BinA_only': 1.9348441362380981, 'fused_vs_marginal_gap': 0.1883237361907959, 'gap_related': 0.1883237361907959, 'gap_unrelated': nan}
    1200 {'loss': 2.7139291763305664, 'nll_fused_A': 1.6519745588302612, 'nll_fused_BinA': 1.6530230045318604, 'nll_marg_A': 1.8843791484832764, 'nll_marg_BinA_only': 1.891041874885559, 'fused_vs_marginal_gap': 0.23801890015602112, 'gap_related': 0.23801890015602112, 'gap_unrelated': nan}
    1300 {'loss': 2.6478285789489746, 'nll_fused_A': 1.6033411026000977, 'nll_fused_BinA': 1.5973707437515259, 'nll_marg_A': 1.8981853723526, 'nll_marg_BinA_only': 1.8843154907226562, 'fused_vs_marginal_gap': 0.28694474697113037, 'gap_related': 0.28694477677345276, 'gap_unrelated': nan}
    1400 {'loss': 2.59649395942688, 'nll_fused_A': 1.5768113136291504, 'nll_fused_BinA': 1.558687448501587, 'nll_marg_A': 1.882543683052063, 'nll_marg_BinA_only': 1.8749048709869385, 'fused_vs_marginal_gap': 0.31621742248535156, 'gap_related': 0.31621742248535156, 'gap_unrelated': nan}
    1500 {'loss': 2.5468544960021973, 'nll_fused_A': 1.5196888446807861, 'nll_fused_BinA': 1.5201923847198486, 'nll_marg_A': 1.9025177955627441, 'nll_marg_BinA_only': 1.8982300758361816, 'fused_vs_marginal_gap': 0.37803757190704346, 'gap_related': 0.37803757190704346, 'gap_unrelated': nan}
    1600 {'loss': 2.430360794067383, 'nll_fused_A': 1.4295095205307007, 'nll_fused_BinA': 1.438987135887146, 'nll_marg_A': 1.875069499015808, 'nll_marg_BinA_only': 1.8765082359313965, 'fused_vs_marginal_gap': 0.43752121925354004, 'gap_related': 0.43752121925354004, 'gap_unrelated': nan}
    1700 {'loss': 2.4318647384643555, 'nll_fused_A': 1.4305706024169922, 'nll_fused_BinA': 1.429113507270813, 'nll_marg_A': 1.9119337797164917, 'nll_marg_BinA_only': 1.9018633365631104, 'fused_vs_marginal_gap': 0.4727499186992645, 'gap_related': 0.47274988889694214, 'gap_unrelated': nan}
    1800 {'loss': 2.379438638687134, 'nll_fused_A': 1.3788354396820068, 'nll_fused_BinA': 1.3955588340759277, 'nll_marg_A': 1.9007635116577148, 'nll_marg_BinA_only': 1.9106943607330322, 'fused_vs_marginal_gap': 0.5151355266571045, 'gap_related': 0.5151355266571045, 'gap_unrelated': nan}
    1900 {'loss': 2.2831053733825684, 'nll_fused_A': 1.3313699960708618, 'nll_fused_BinA': 1.3332562446594238, 'nll_marg_A': 1.8347935676574707, 'nll_marg_BinA_only': 1.8389661312103271, 'fused_vs_marginal_gap': 0.5057098865509033, 'gap_related': 0.5057098865509033, 'gap_unrelated': nan}
    2000 {'loss': 2.318894386291504, 'nll_fused_A': 1.3520007133483887, 'nll_fused_BinA': 1.3484078645706177, 'nll_marg_A': 1.8829538822174072, 'nll_marg_BinA_only': 1.8860408067703247, 'fused_vs_marginal_gap': 0.5376328229904175, 'gap_related': 0.5376328229904175, 'gap_unrelated': nan}
    2100 {'loss': 2.2365448474884033, 'nll_fused_A': 1.2844852209091187, 'nll_fused_BinA': 1.2917625904083252, 'nll_marg_A': 1.8647888898849487, 'nll_marg_BinA_only': 1.8745534420013428, 'fused_vs_marginal_gap': 0.5827908515930176, 'gap_related': 0.5827908515930176, 'gap_unrelated': nan}
    2200 {'loss': 2.251328229904175, 'nll_fused_A': 1.2936062812805176, 'nll_fused_BinA': 1.3066716194152832, 'nll_marg_A': 1.8552488088607788, 'nll_marg_BinA_only': 1.855283260345459, 'fused_vs_marginal_gap': 0.5486115217208862, 'gap_related': 0.5486115217208862, 'gap_unrelated': nan}
    2300 {'loss': 2.286069393157959, 'nll_fused_A': 1.323176622390747, 'nll_fused_BinA': 1.3178186416625977, 'nll_marg_A': 1.904326319694519, 'nll_marg_BinA_only': 1.8872509002685547, 'fused_vs_marginal_gap': 0.569432258605957, 'gap_related': 0.569432258605957, 'gap_unrelated': nan}
    2400 {'loss': 2.2403647899627686, 'nll_fused_A': 1.297208547592163, 'nll_fused_BinA': 1.2832103967666626, 'nll_marg_A': 1.893305778503418, 'nll_marg_BinA_only': 1.8844072818756104, 'fused_vs_marginal_gap': 0.6011967658996582, 'gap_related': 0.6011967658996582, 'gap_unrelated': nan}
    2500 {'loss': 2.0721287727355957, 'nll_fused_A': 1.1725493669509888, 'nll_fused_BinA': 1.1627386808395386, 'nll_marg_A': 1.858750820159912, 'nll_marg_BinA_only': 1.8568556308746338, 'fused_vs_marginal_gap': 0.6941168308258057, 'gap_related': 0.6941168308258057, 'gap_unrelated': nan}
    2600 {'loss': 2.157985210418701, 'nll_fused_A': 1.2209844589233398, 'nll_fused_BinA': 1.2337324619293213, 'nll_marg_A': 1.8598580360412598, 'nll_marg_BinA_only': 1.8739213943481445, 'fused_vs_marginal_gap': 0.6401888132095337, 'gap_related': 0.6401888132095337, 'gap_unrelated': nan}
    2700 {'loss': 2.0507662296295166, 'nll_fused_A': 1.143157720565796, 'nll_fused_BinA': 1.1506764888763428, 'nll_marg_A': 1.8571412563323975, 'nll_marg_BinA_only': 1.860252022743225, 'fused_vs_marginal_gap': 0.7095755338668823, 'gap_related': 0.7095755338668823, 'gap_unrelated': nan}
    2800 {'loss': 2.0806384086608887, 'nll_fused_A': 1.1798746585845947, 'nll_fused_BinA': 1.1648180484771729, 'nll_marg_A': 1.8728597164154053, 'nll_marg_BinA_only': 1.8691868782043457, 'fused_vs_marginal_gap': 0.7043689489364624, 'gap_related': 0.7043689489364624, 'gap_unrelated': nan}
    2900 {'loss': 2.0230300426483154, 'nll_fused_A': 1.121490240097046, 'nll_fused_BinA': 1.1335725784301758, 'nll_marg_A': 1.843367576599121, 'nll_marg_BinA_only': 1.8429837226867676, 'fused_vs_marginal_gap': 0.7094112634658813, 'gap_related': 0.7094112634658813, 'gap_unrelated': nan}
    3000 {'loss': 2.037912368774414, 'nll_fused_A': 1.1387858390808105, 'nll_fused_BinA': 1.1466695070266724, 'nll_marg_A': 1.832023024559021, 'nll_marg_BinA_only': 1.831141471862793, 'fused_vs_marginal_gap': 0.684471845626831, 'gap_related': 0.684471845626831, 'gap_unrelated': nan}
    3100 {'loss': 2.072016716003418, 'nll_fused_A': 1.161365270614624, 'nll_fused_BinA': 1.169061541557312, 'nll_marg_A': 1.8484851121902466, 'nll_marg_BinA_only': 1.8600869178771973, 'fused_vs_marginal_gap': 0.6910252571105957, 'gap_related': 0.6910252571105957, 'gap_unrelated': nan}
    3200 {'loss': 2.096480369567871, 'nll_fused_A': 1.1891734600067139, 'nll_fused_BinA': 1.184098482131958, 'nll_marg_A': 1.8520996570587158, 'nll_marg_BinA_only': 1.851860523223877, 'fused_vs_marginal_gap': 0.6677622199058533, 'gap_related': 0.667762279510498, 'gap_unrelated': nan}
    3300 {'loss': 2.0345449447631836, 'nll_fused_A': 1.1489170789718628, 'nll_fused_BinA': 1.1368582248687744, 'nll_marg_A': 1.843371868133545, 'nll_marg_BinA_only': 1.838014841079712, 'fused_vs_marginal_gap': 0.7011565566062927, 'gap_related': 0.7011565566062927, 'gap_unrelated': nan}
    3400 {'loss': 1.9987003803253174, 'nll_fused_A': 1.1066763401031494, 'nll_fused_BinA': 1.1160542964935303, 'nll_marg_A': 1.8354768753051758, 'nll_marg_BinA_only': 1.8354737758636475, 'fused_vs_marginal_gap': 0.7194195985794067, 'gap_related': 0.7194195985794067, 'gap_unrelated': nan}
    3500 {'loss': 1.9890177249908447, 'nll_fused_A': 1.1102943420410156, 'nll_fused_BinA': 1.1051278114318848, 'nll_marg_A': 1.8360052108764648, 'nll_marg_BinA_only': 1.8371729850769043, 'fused_vs_marginal_gap': 0.7320452928543091, 'gap_related': 0.7320452928543091, 'gap_unrelated': nan}
    3600 {'loss': 2.048435926437378, 'nll_fused_A': 1.1522212028503418, 'nll_fused_BinA': 1.1560989618301392, 'nll_marg_A': 1.822235107421875, 'nll_marg_BinA_only': 1.8214666843414307, 'fused_vs_marginal_gap': 0.6653677225112915, 'gap_related': 0.6653677225112915, 'gap_unrelated': nan}
    3700 {'loss': 1.889390468597412, 'nll_fused_A': 1.0276223421096802, 'nll_fused_BinA': 1.0305900573730469, 'nll_marg_A': 1.8350458145141602, 'nll_marg_BinA_only': 1.8358170986175537, 'fused_vs_marginal_gap': 0.8052270412445068, 'gap_related': 0.8052271008491516, 'gap_unrelated': nan}
    3800 {'loss': 1.9390161037445068, 'nll_fused_A': 1.0710433721542358, 'nll_fused_BinA': 1.0720205307006836, 'nll_marg_A': 1.8189417123794556, 'nll_marg_BinA_only': 1.813170075416565, 'fused_vs_marginal_gap': 0.7411496639251709, 'gap_related': 0.7411496639251709, 'gap_unrelated': nan}
    3900 {'loss': 2.0393104553222656, 'nll_fused_A': 1.1382980346679688, 'nll_fused_BinA': 1.1315724849700928, 'nll_marg_A': 1.8874952793121338, 'nll_marg_BinA_only': 1.8905975818634033, 'fused_vs_marginal_gap': 0.7590250968933105, 'gap_related': 0.7590250968933105, 'gap_unrelated': nan}
    4000 {'loss': 2.014055013656616, 'nll_fused_A': 1.1321289539337158, 'nll_fused_BinA': 1.1165461540222168, 'nll_marg_A': 1.8595671653747559, 'nll_marg_BinA_only': 1.8653497695922852, 'fused_vs_marginal_gap': 0.7488036155700684, 'gap_related': 0.7488036155700684, 'gap_unrelated': nan}
    4100 {'loss': 2.007847547531128, 'nll_fused_A': 1.119112253189087, 'nll_fused_BinA': 1.1238057613372803, 'nll_marg_A': 1.8276934623718262, 'nll_marg_BinA_only': 1.8269445896148682, 'fused_vs_marginal_gap': 0.7031388282775879, 'gap_related': 0.7031388282775879, 'gap_unrelated': nan}
    4200 {'loss': 1.885051965713501, 'nll_fused_A': 1.0205599069595337, 'nll_fused_BinA': 1.0331995487213135, 'nll_marg_A': 1.8189479112625122, 'nll_marg_BinA_only': 1.8211579322814941, 'fused_vs_marginal_gap': 0.7879583835601807, 'gap_related': 0.7879583835601807, 'gap_unrelated': nan}
    4300 {'loss': 1.991352915763855, 'nll_fused_A': 1.097153663635254, 'nll_fused_BinA': 1.1156405210494995, 'nll_marg_A': 1.821887493133545, 'nll_marg_BinA_only': 1.8248858451843262, 'fused_vs_marginal_gap': 0.7092453241348267, 'gap_related': 0.7092453837394714, 'gap_unrelated': nan}
    4400 {'loss': 1.837799072265625, 'nll_fused_A': 1.002399206161499, 'nll_fused_BinA': 1.0058115720748901, 'nll_marg_A': 1.7708925008773804, 'nll_marg_BinA_only': 1.7795078754425049, 'fused_vs_marginal_gap': 0.7736961841583252, 'gap_related': 0.7736961841583252, 'gap_unrelated': nan}
    4500 {'loss': 1.8866710662841797, 'nll_fused_A': 1.0352741479873657, 'nll_fused_BinA': 1.0374494791030884, 'nll_marg_A': 1.7954645156860352, 'nll_marg_BinA_only': 1.805171251296997, 'fused_vs_marginal_gap': 0.7677217125892639, 'gap_related': 0.7677216529846191, 'gap_unrelated': nan}
    4600 {'loss': 1.8975257873535156, 'nll_fused_A': 1.0507508516311646, 'nll_fused_BinA': 1.0354299545288086, 'nll_marg_A': 1.8229016065597534, 'nll_marg_BinA_only': 1.8255884647369385, 'fused_vs_marginal_gap': 0.7901586294174194, 'gap_related': 0.7901586294174194, 'gap_unrelated': nan}
    4700 {'loss': 1.8695502281188965, 'nll_fused_A': 1.0284013748168945, 'nll_fused_BinA': 1.0189337730407715, 'nll_marg_A': 1.806986927986145, 'nll_marg_BinA_only': 1.808081030845642, 'fused_vs_marginal_gap': 0.789147138595581, 'gap_related': 0.789147138595581, 'gap_unrelated': nan}
    4800 {'loss': 1.8036417961120605, 'nll_fused_A': 0.966354250907898, 'nll_fused_BinA': 0.9801677465438843, 'nll_marg_A': 1.7785593271255493, 'nll_marg_BinA_only': 1.7806066274642944, 'fused_vs_marginal_gap': 0.8004387617111206, 'gap_related': 0.8004387617111206, 'gap_unrelated': nan}
    4900 {'loss': 1.7896778583526611, 'nll_fused_A': 0.9694873094558716, 'nll_fused_BinA': 0.9555410742759705, 'nll_marg_A': 1.8109683990478516, 'nll_marg_BinA_only': 1.797804355621338, 'fused_vs_marginal_gap': 0.8422631621360779, 'gap_related': 0.8422631621360779, 'gap_unrelated': nan}
    5000 {'loss': 1.856410264968872, 'nll_fused_A': 1.01539146900177, 'nll_fused_BinA': 1.004908561706543, 'nll_marg_A': 1.8229471445083618, 'nll_marg_BinA_only': 1.8133281469345093, 'fused_vs_marginal_gap': 0.8084194660186768, 'gap_related': 0.8084194660186768, 'gap_unrelated': nan}
    5100 {'loss': 1.8616306781768799, 'nll_fused_A': 1.0213316679000854, 'nll_fused_BinA': 1.0134775638580322, 'nll_marg_A': 1.8058452606201172, 'nll_marg_BinA_only': 1.7941319942474365, 'fused_vs_marginal_gap': 0.7806543111801147, 'gap_related': 0.7806543111801147, 'gap_unrelated': nan}
    5200 {'loss': 1.825242519378662, 'nll_fused_A': 0.9899594187736511, 'nll_fused_BinA': 0.9946534037590027, 'nll_marg_A': 1.7786705493927002, 'nll_marg_BinA_only': 1.7859997749328613, 'fused_vs_marginal_gap': 0.7913464307785034, 'gap_related': 0.7913464307785034, 'gap_unrelated': nan}
    5300 {'loss': 1.747209072113037, 'nll_fused_A': 0.9399073123931885, 'nll_fused_BinA': 0.9329686164855957, 'nll_marg_A': 1.7742276191711426, 'nll_marg_BinA_only': 1.769474983215332, 'fused_vs_marginal_gap': 0.8365063667297363, 'gap_related': 0.8365063667297363, 'gap_unrelated': nan}
    5400 {'loss': 1.7532129287719727, 'nll_fused_A': 0.9355889558792114, 'nll_fused_BinA': 0.93800950050354, 'nll_marg_A': 1.7817559242248535, 'nll_marg_BinA_only': 1.7806869745254517, 'fused_vs_marginal_gap': 0.8426775932312012, 'gap_related': 0.8426775336265564, 'gap_unrelated': nan}
    5500 {'loss': 1.7901456356048584, 'nll_fused_A': 0.9602181315422058, 'nll_fused_BinA': 0.9553775787353516, 'nll_marg_A': 1.8223416805267334, 'nll_marg_BinA_only': 1.8142824172973633, 'fused_vs_marginal_gap': 0.8589046597480774, 'gap_related': 0.8589047193527222, 'gap_unrelated': nan}
    5600 {'loss': 1.7733709812164307, 'nll_fused_A': 0.9681535363197327, 'nll_fused_BinA': 0.9598634839057922, 'nll_marg_A': 1.7435379028320312, 'nll_marg_BinA_only': 1.7434749603271484, 'fused_vs_marginal_gap': 0.783611536026001, 'gap_related': 0.783611536026001, 'gap_unrelated': nan}
    5700 {'loss': 1.7493479251861572, 'nll_fused_A': 0.9222118854522705, 'nll_fused_BinA': 0.9318063259124756, 'nll_marg_A': 1.802926778793335, 'nll_marg_BinA_only': 1.808844804763794, 'fused_vs_marginal_gap': 0.8770385980606079, 'gap_related': 0.8770385980606079, 'gap_unrelated': nan}
    5800 {'loss': 1.7075809240341187, 'nll_fused_A': 0.9154691696166992, 'nll_fused_BinA': 0.9130851030349731, 'nll_marg_A': 1.7328500747680664, 'nll_marg_BinA_only': 1.7319728136062622, 'fused_vs_marginal_gap': 0.8188876509666443, 'gap_related': 0.8188877105712891, 'gap_unrelated': nan}
    5900 {'loss': 1.7642278671264648, 'nll_fused_A': 0.9392637014389038, 'nll_fused_BinA': 0.9426329135894775, 'nll_marg_A': 1.7993862628936768, 'nll_marg_BinA_only': 1.795573115348816, 'fused_vs_marginal_gap': 0.8529402017593384, 'gap_related': 0.8529402017593384, 'gap_unrelated': nan}
    6000 {'loss': 1.795518159866333, 'nll_fused_A': 0.9758938550949097, 'nll_fused_BinA': 0.9772907495498657, 'nll_marg_A': 1.751530647277832, 'nll_marg_BinA_only': 1.7512450218200684, 'fused_vs_marginal_gap': 0.7739541530609131, 'gap_related': 0.7739541530609131, 'gap_unrelated': nan}
    6100 {'loss': 1.6605372428894043, 'nll_fused_A': 0.877274751663208, 'nll_fused_BinA': 0.8792400360107422, 'nll_marg_A': 1.7270493507385254, 'nll_marg_BinA_only': 1.7262141704559326, 'fused_vs_marginal_gap': 0.8469741344451904, 'gap_related': 0.8469741344451904, 'gap_unrelated': nan}
    6200 {'loss': 1.7787810564041138, 'nll_fused_A': 0.9680410027503967, 'nll_fused_BinA': 0.9610744714736938, 'nll_marg_A': 1.7576475143432617, 'nll_marg_BinA_only': 1.761091947555542, 'fused_vs_marginal_gap': 0.8000175952911377, 'gap_related': 0.8000175952911377, 'gap_unrelated': nan}
    6300 {'loss': 1.6058282852172852, 'nll_fused_A': 0.840608537197113, 'nll_fused_BinA': 0.839188814163208, 'nll_marg_A': 1.7148561477661133, 'nll_marg_BinA_only': 1.7072737216949463, 'fused_vs_marginal_gap': 0.8680849075317383, 'gap_related': 0.8680849075317383, 'gap_unrelated': nan}
    6400 {'loss': 1.7375702857971191, 'nll_fused_A': 0.9256937503814697, 'nll_fused_BinA': 0.9256291389465332, 'nll_marg_A': 1.780776858329773, 'nll_marg_BinA_only': 1.7694599628448486, 'fused_vs_marginal_gap': 0.8438308238983154, 'gap_related': 0.8438308238983154, 'gap_unrelated': nan}
    6500 {'loss': 1.7401537895202637, 'nll_fused_A': 0.923342227935791, 'nll_fused_BinA': 0.9256685972213745, 'nll_marg_A': 1.7916083335876465, 'nll_marg_BinA_only': 1.785206913948059, 'fused_vs_marginal_gap': 0.8595383167266846, 'gap_related': 0.8595382571220398, 'gap_unrelated': nan}
    6600 {'loss': 1.7559514045715332, 'nll_fused_A': 0.9432767629623413, 'nll_fused_BinA': 0.9451321363449097, 'nll_marg_A': 1.7594537734985352, 'nll_marg_BinA_only': 1.7541301250457764, 'fused_vs_marginal_gap': 0.8089978694915771, 'gap_related': 0.8089978694915771, 'gap_unrelated': nan}
    6700 {'loss': 1.6688411235809326, 'nll_fused_A': 0.8937181234359741, 'nll_fused_BinA': 0.8919593095779419, 'nll_marg_A': 1.695887804031372, 'nll_marg_BinA_only': 1.694362998008728, 'fused_vs_marginal_gap': 0.8024038076400757, 'gap_related': 0.8024038076400757, 'gap_unrelated': nan}
    6800 {'loss': 1.6951675415039062, 'nll_fused_A': 0.8987150192260742, 'nll_fused_BinA': 0.9022992849349976, 'nll_marg_A': 1.7441792488098145, 'nll_marg_BinA_only': 1.7490321397781372, 'fused_vs_marginal_gap': 0.8467328548431396, 'gap_related': 0.8467328548431396, 'gap_unrelated': nan}
    6900 {'loss': 1.7249674797058105, 'nll_fused_A': 0.9140197038650513, 'nll_fused_BinA': 0.9135041236877441, 'nll_marg_A': 1.7908579111099243, 'nll_marg_BinA_only': 1.7810312509536743, 'fused_vs_marginal_gap': 0.8675270676612854, 'gap_related': 0.8675270080566406, 'gap_unrelated': nan}
    7000 {'loss': 1.6210176944732666, 'nll_fused_A': 0.8620808720588684, 'nll_fused_BinA': 0.8492774963378906, 'nll_marg_A': 1.7103865146636963, 'nll_marg_BinA_only': 1.7080445289611816, 'fused_vs_marginal_gap': 0.8587669730186462, 'gap_related': 0.8587669730186462, 'gap_unrelated': nan}
    7100 {'loss': 1.6285734176635742, 'nll_fused_A': 0.8567572236061096, 'nll_fused_BinA': 0.8474818468093872, 'nll_marg_A': 1.746881127357483, 'nll_marg_BinA_only': 1.7509782314300537, 'fused_vs_marginal_gap': 0.903496503829956, 'gap_related': 0.9034964442253113, 'gap_unrelated': nan}
    7200 {'loss': 1.657344102859497, 'nll_fused_A': 0.8899447917938232, 'nll_fused_BinA': 0.8681947588920593, 'nll_marg_A': 1.7405526638031006, 'nll_marg_BinA_only': 1.7267708778381348, 'fused_vs_marginal_gap': 0.8585761189460754, 'gap_related': 0.8585760593414307, 'gap_unrelated': nan}
    7300 {'loss': 1.582922101020813, 'nll_fused_A': 0.8181285262107849, 'nll_fused_BinA': 0.8267993927001953, 'nll_marg_A': 1.7022804021835327, 'nll_marg_BinA_only': 1.7002589702606201, 'fused_vs_marginal_gap': 0.87345951795578, 'gap_related': 0.8734595775604248, 'gap_unrelated': nan}
    7400 {'loss': 1.6439331769943237, 'nll_fused_A': 0.8485658168792725, 'nll_fused_BinA': 0.8672449588775635, 'nll_marg_A': 1.740394949913025, 'nll_marg_BinA_only': 1.7361825704574585, 'fused_vs_marginal_gap': 0.868937611579895, 'gap_related': 0.868937611579895, 'gap_unrelated': nan}
    7500 {'loss': 1.6138906478881836, 'nll_fused_A': 0.8460222482681274, 'nll_fused_BinA': 0.8390335440635681, 'nll_marg_A': 1.7368342876434326, 'nll_marg_BinA_only': 1.7423443794250488, 'fused_vs_marginal_gap': 0.9033108353614807, 'gap_related': 0.9033107757568359, 'gap_unrelated': nan}
    7600 {'loss': 1.5992532968521118, 'nll_fused_A': 0.8244428634643555, 'nll_fused_BinA': 0.8337821960449219, 'nll_marg_A': 1.7271273136138916, 'nll_marg_BinA_only': 1.73465096950531, 'fused_vs_marginal_gap': 0.9008687734603882, 'gap_related': 0.9008687734603882, 'gap_unrelated': nan}
    7700 {'loss': 1.5941691398620605, 'nll_fused_A': 0.8308149576187134, 'nll_fused_BinA': 0.8218975067138672, 'nll_marg_A': 1.7434239387512207, 'nll_marg_BinA_only': 1.7343182563781738, 'fused_vs_marginal_gap': 0.9124207496643066, 'gap_related': 0.9124207496643066, 'gap_unrelated': nan}
    7800 {'loss': 1.61430025100708, 'nll_fused_A': 0.8429937362670898, 'nll_fused_BinA': 0.8384689688682556, 'nll_marg_A': 1.7431107759475708, 'nll_marg_BinA_only': 1.7376437187194824, 'fused_vs_marginal_gap': 0.899174690246582, 'gap_related': 0.899174690246582, 'gap_unrelated': nan}
    7900 {'loss': 1.5635069608688354, 'nll_fused_A': 0.8184802532196045, 'nll_fused_BinA': 0.811508297920227, 'nll_marg_A': 1.6881818771362305, 'nll_marg_BinA_only': 1.6844546794891357, 'fused_vs_marginal_gap': 0.8729462623596191, 'gap_related': 0.8729462623596191, 'gap_unrelated': nan}
    8000 {'loss': 1.6652898788452148, 'nll_fused_A': 0.871787428855896, 'nll_fused_BinA': 0.8713600039482117, 'nll_marg_A': 1.7746455669403076, 'nll_marg_BinA_only': 1.7856096029281616, 'fused_vs_marginal_gap': 0.9142496585845947, 'gap_related': 0.9142496585845947, 'gap_unrelated': nan}
    8100 {'loss': 1.5036020278930664, 'nll_fused_A': 0.7676342725753784, 'nll_fused_BinA': 0.7751495838165283, 'nll_marg_A': 1.6605405807495117, 'nll_marg_BinA_only': 1.6557226181030273, 'fused_vs_marginal_gap': 0.8805731534957886, 'gap_related': 0.8805730938911438, 'gap_unrelated': nan}
    8200 {'loss': 1.576276183128357, 'nll_fused_A': 0.8141025304794312, 'nll_fused_BinA': 0.8204957246780396, 'nll_marg_A': 1.7051656246185303, 'nll_marg_BinA_only': 1.710824728012085, 'fused_vs_marginal_gap': 0.8903290033340454, 'gap_related': 0.8903289437294006, 'gap_unrelated': nan}
    8300 {'loss': 1.684119701385498, 'nll_fused_A': 0.8890858292579651, 'nll_fused_BinA': 0.8859694600105286, 'nll_marg_A': 1.7714147567749023, 'nll_marg_BinA_only': 1.7504106760025024, 'fused_vs_marginal_gap': 0.8644412159919739, 'gap_related': 0.8644412159919739, 'gap_unrelated': nan}
    8400 {'loss': 1.5614964962005615, 'nll_fused_A': 0.800977349281311, 'nll_fused_BinA': 0.8087193965911865, 'nll_marg_A': 1.7082793712615967, 'nll_marg_BinA_only': 1.7213693857192993, 'fused_vs_marginal_gap': 0.9126499891281128, 'gap_related': 0.9126499891281128, 'gap_unrelated': nan}
    8500 {'loss': 1.4844095706939697, 'nll_fused_A': 0.7579959630966187, 'nll_fused_BinA': 0.7533724308013916, 'nll_marg_A': 1.678794503211975, 'nll_marg_BinA_only': 1.6785417795181274, 'fused_vs_marginal_gap': 0.9251693487167358, 'gap_related': 0.9251693487167358, 'gap_unrelated': nan}
    8600 {'loss': 1.5130678415298462, 'nll_fused_A': 0.7776550054550171, 'nll_fused_BinA': 0.7711830139160156, 'nll_marg_A': 1.6952942609786987, 'nll_marg_BinA_only': 1.7011600732803345, 'fused_vs_marginal_gap': 0.9299770593643188, 'gap_related': 0.9299770593643188, 'gap_unrelated': nan}
    8700 {'loss': 1.5832924842834473, 'nll_fused_A': 0.8220441341400146, 'nll_fused_BinA': 0.8262768983840942, 'nll_marg_A': 1.7013410329818726, 'nll_marg_BinA_only': 1.689338207244873, 'fused_vs_marginal_gap': 0.863061249256134, 'gap_related': 0.863061249256134, 'gap_unrelated': nan}
    8800 {'loss': 1.5291826725006104, 'nll_fused_A': 0.7893248796463013, 'nll_fused_BinA': 0.7808146476745605, 'nll_marg_A': 1.705235481262207, 'nll_marg_BinA_only': 1.6904693841934204, 'fused_vs_marginal_gap': 0.9096547365188599, 'gap_related': 0.9096547365188599, 'gap_unrelated': nan}
    8900 {'loss': 1.6160600185394287, 'nll_fused_A': 0.8364210724830627, 'nll_fused_BinA': 0.8487073183059692, 'nll_marg_A': 1.7214208841323853, 'nll_marg_BinA_only': 1.708857536315918, 'fused_vs_marginal_gap': 0.8601502180099487, 'gap_related': 0.860150158405304, 'gap_unrelated': nan}
    9000 {'loss': 1.6263225078582764, 'nll_fused_A': 0.8526135683059692, 'nll_fused_BinA': 0.8533364534378052, 'nll_marg_A': 1.7240064144134521, 'nll_marg_BinA_only': 1.7344120740890503, 'fused_vs_marginal_gap': 0.8810755610466003, 'gap_related': 0.8810756206512451, 'gap_unrelated': nan}
    9100 {'loss': 1.540252923965454, 'nll_fused_A': 0.8229209184646606, 'nll_fused_BinA': 0.7988896369934082, 'nll_marg_A': 1.6482899188995361, 'nll_marg_BinA_only': 1.6521432399749756, 'fused_vs_marginal_gap': 0.8532536029815674, 'gap_related': 0.8532536029815674, 'gap_unrelated': nan}
    9200 {'loss': 1.5047136545181274, 'nll_fused_A': 0.7735137939453125, 'nll_fused_BinA': 0.7699077129364014, 'nll_marg_A': 1.6758391857147217, 'nll_marg_BinA_only': 1.6793044805526733, 'fused_vs_marginal_gap': 0.909396767616272, 'gap_related': 0.909396767616272, 'gap_unrelated': nan}
    9300 {'loss': 1.4536821842193604, 'nll_fused_A': 0.7242588996887207, 'nll_fused_BinA': 0.7311135530471802, 'nll_marg_A': 1.6843032836914062, 'nll_marg_BinA_only': 1.691148042678833, 'fused_vs_marginal_gap': 0.9600344896316528, 'gap_related': 0.9600344896316528, 'gap_unrelated': nan}
    9400 {'loss': 1.4493459463119507, 'nll_fused_A': 0.738368034362793, 'nll_fused_BinA': 0.7339588403701782, 'nll_marg_A': 1.646255373954773, 'nll_marg_BinA_only': 1.6461739540100098, 'fused_vs_marginal_gap': 0.9122151136398315, 'gap_related': 0.9122151136398315, 'gap_unrelated': nan}
    9500 {'loss': 1.4679064750671387, 'nll_fused_A': 0.7362622022628784, 'nll_fused_BinA': 0.7430164813995361, 'nll_marg_A': 1.680037498474121, 'nll_marg_BinA_only': 1.6878600120544434, 'fused_vs_marginal_gap': 0.9448435306549072, 'gap_related': 0.9448435306549072, 'gap_unrelated': nan}
    9600 {'loss': 1.547623872756958, 'nll_fused_A': 0.8162136077880859, 'nll_fused_BinA': 0.8134583234786987, 'nll_marg_A': 1.6310046911239624, 'nll_marg_BinA_only': 1.6431515216827393, 'fused_vs_marginal_gap': 0.8296931982040405, 'gap_related': 0.8296931982040405, 'gap_unrelated': nan}
    9700 {'loss': 1.5092684030532837, 'nll_fused_A': 0.7712416648864746, 'nll_fused_BinA': 0.7825378179550171, 'nll_marg_A': 1.651193618774414, 'nll_marg_BinA_only': 1.6555031538009644, 'fused_vs_marginal_gap': 0.8729653358459473, 'gap_related': 0.8729653358459473, 'gap_unrelated': nan}
    9800 {'loss': 1.5495946407318115, 'nll_fused_A': 0.7937144041061401, 'nll_fused_BinA': 0.804821252822876, 'nll_marg_A': 1.6888632774353027, 'nll_marg_BinA_only': 1.6847288608551025, 'fused_vs_marginal_gap': 0.8799075484275818, 'gap_related': 0.879907488822937, 'gap_unrelated': nan}
    9900 {'loss': 1.407773733139038, 'nll_fused_A': 0.7093894481658936, 'nll_fused_BinA': 0.6906096935272217, 'nll_marg_A': 1.681157112121582, 'nll_marg_BinA_only': 1.6700012683868408, 'fused_vs_marginal_gap': 0.9793914556503296, 'gap_related': 0.9793914556503296, 'gap_unrelated': nan}
    10000 {'loss': 1.5107091665267944, 'nll_fused_A': 0.7695440053939819, 'nll_fused_BinA': 0.7754323482513428, 'nll_marg_A': 1.681378722190857, 'nll_marg_BinA_only': 1.6778578758239746, 'fused_vs_marginal_gap': 0.9024255871772766, 'gap_related': 0.9024255871772766, 'gap_unrelated': nan}
    10100 {'loss': 1.5755481719970703, 'nll_fused_A': 0.8153709173202515, 'nll_fused_BinA': 0.8135533332824707, 'nll_marg_A': 1.724611759185791, 'nll_marg_BinA_only': 1.7250233888626099, 'fused_vs_marginal_gap': 0.9114700555801392, 'gap_related': 0.9114700555801392, 'gap_unrelated': nan}
    10200 {'loss': 1.430204153060913, 'nll_fused_A': 0.7262812852859497, 'nll_fused_BinA': 0.7127228379249573, 'nll_marg_A': 1.6653231382369995, 'nll_marg_BinA_only': 1.6505064964294434, 'fused_vs_marginal_gap': 0.9377837181091309, 'gap_related': 0.9377836585044861, 'gap_unrelated': nan}
    10300 {'loss': 1.330604910850525, 'nll_fused_A': 0.654765784740448, 'nll_fused_BinA': 0.6482017040252686, 'nll_marg_A': 1.6199114322662354, 'nll_marg_BinA_only': 1.6210029125213623, 'fused_vs_marginal_gap': 0.9728013277053833, 'gap_related': 0.9728012681007385, 'gap_unrelated': nan}
    10400 {'loss': 1.5096807479858398, 'nll_fused_A': 0.7821756601333618, 'nll_fused_BinA': 0.7770821452140808, 'nll_marg_A': 1.6598196029663086, 'nll_marg_BinA_only': 1.6529395580291748, 'fused_vs_marginal_gap': 0.8758572340011597, 'gap_related': 0.8758572340011597, 'gap_unrelated': nan}
    10500 {'loss': 1.3949074745178223, 'nll_fused_A': 0.6931366920471191, 'nll_fused_BinA': 0.6893153190612793, 'nll_marg_A': 1.658836841583252, 'nll_marg_BinA_only': 1.6652717590332031, 'fused_vs_marginal_gap': 0.9759564399719238, 'gap_related': 0.975956380367279, 'gap_unrelated': nan}
    10600 {'loss': 1.3809881210327148, 'nll_fused_A': 0.693058967590332, 'nll_fused_BinA': 0.6859481930732727, 'nll_marg_A': 1.6237406730651855, 'nll_marg_BinA_only': 1.6186028718948364, 'fused_vs_marginal_gap': 0.9326547384262085, 'gap_related': 0.9326547384262085, 'gap_unrelated': nan}
    10700 {'loss': 1.3905774354934692, 'nll_fused_A': 0.7043533325195312, 'nll_fused_BinA': 0.6847188472747803, 'nll_marg_A': 1.6485084295272827, 'nll_marg_BinA_only': 1.6451911926269531, 'fused_vs_marginal_gap': 0.9604723453521729, 'gap_related': 0.9604722857475281, 'gap_unrelated': nan}
    10800 {'loss': 1.4378466606140137, 'nll_fused_A': 0.7406796216964722, 'nll_fused_BinA': 0.7219310402870178, 'nll_marg_A': 1.6457056999206543, 'nll_marg_BinA_only': 1.6416583061218262, 'fused_vs_marginal_gap': 0.9197273254394531, 'gap_related': 0.9197273254394531, 'gap_unrelated': nan}
    10900 {'loss': 1.4529531002044678, 'nll_fused_A': 0.7355376482009888, 'nll_fused_BinA': 0.7306223511695862, 'nll_marg_A': 1.672231674194336, 'nll_marg_BinA_only': 1.6660048961639404, 'fused_vs_marginal_gap': 0.9353824257850647, 'gap_related': 0.9353824257850647, 'gap_unrelated': nan}
    11000 {'loss': 1.3266302347183228, 'nll_fused_A': 0.6519891023635864, 'nll_fused_BinA': 0.6502436995506287, 'nll_marg_A': 1.6026326417922974, 'nll_marg_BinA_only': 1.6007108688354492, 'fused_vs_marginal_gap': 0.9504672288894653, 'gap_related': 0.9504672288894653, 'gap_unrelated': nan}
    11100 {'loss': 1.3716015815734863, 'nll_fused_A': 0.6768344044685364, 'nll_fused_BinA': 0.6826421618461609, 'nll_marg_A': 1.619696855545044, 'nll_marg_BinA_only': 1.615239143371582, 'fused_vs_marginal_gap': 0.9325970411300659, 'gap_related': 0.9325970411300659, 'gap_unrelated': nan}
    11200 {'loss': 1.3528602123260498, 'nll_fused_A': 0.6631238460540771, 'nll_fused_BinA': 0.6668556928634644, 'nll_marg_A': 1.6235575675964355, 'nll_marg_BinA_only': 1.6315267086029053, 'fused_vs_marginal_gap': 0.9646711349487305, 'gap_related': 0.9646711349487305, 'gap_unrelated': nan}
    11300 {'loss': 1.4253058433532715, 'nll_fused_A': 0.7258844971656799, 'nll_fused_BinA': 0.7118683457374573, 'nll_marg_A': 1.6522407531738281, 'nll_marg_BinA_only': 1.6617605686187744, 'fused_vs_marginal_gap': 0.9498922824859619, 'gap_related': 0.9498922824859619, 'gap_unrelated': nan}
    11400 {'loss': 1.4307365417480469, 'nll_fused_A': 0.7335765361785889, 'nll_fused_BinA': 0.7227917909622192, 'nll_marg_A': 1.6262391805648804, 'nll_marg_BinA_only': 1.6407999992370605, 'fused_vs_marginal_gap': 0.9180082082748413, 'gap_related': 0.9180082082748413, 'gap_unrelated': nan}
    11500 {'loss': 1.3904824256896973, 'nll_fused_A': 0.6973302960395813, 'nll_fused_BinA': 0.6949516534805298, 'nll_marg_A': 1.621105670928955, 'nll_marg_BinA_only': 1.627441167831421, 'fused_vs_marginal_gap': 0.9324895739555359, 'gap_related': 0.9324895739555359, 'gap_unrelated': nan}
    11600 {'loss': 1.4186179637908936, 'nll_fused_A': 0.6956658363342285, 'nll_fused_BinA': 0.7138391733169556, 'nll_marg_A': 1.6535966396331787, 'nll_marg_BinA_only': 1.659275770187378, 'fused_vs_marginal_gap': 0.9454365372657776, 'gap_related': 0.9454365372657776, 'gap_unrelated': nan}
    11700 {'loss': 1.3715605735778809, 'nll_fused_A': 0.6852555274963379, 'nll_fused_BinA': 0.6758127212524414, 'nll_marg_A': 1.633903980255127, 'nll_marg_BinA_only': 1.6245180368423462, 'fused_vs_marginal_gap': 0.9487053155899048, 'gap_related': 0.9487053155899048, 'gap_unrelated': nan}
    11800 {'loss': 1.3155359029769897, 'nll_fused_A': 0.6447409391403198, 'nll_fused_BinA': 0.648622989654541, 'nll_marg_A': 1.5783019065856934, 'nll_marg_BinA_only': 1.5787482261657715, 'fused_vs_marginal_gap': 0.9301252365112305, 'gap_related': 0.9301252961158752, 'gap_unrelated': nan}
    11900 {'loss': 1.3171718120574951, 'nll_fused_A': 0.6533106565475464, 'nll_fused_BinA': 0.6349459290504456, 'nll_marg_A': 1.6207756996154785, 'nll_marg_BinA_only': 1.6135859489440918, 'fused_vs_marginal_gap': 0.9786399602890015, 'gap_related': 0.9786400198936462, 'gap_unrelated': nan}
    12000 {'loss': 1.3559679985046387, 'nll_fused_A': 0.6785426735877991, 'nll_fused_BinA': 0.6771738529205322, 'nll_marg_A': 1.584104299545288, 'nll_marg_BinA_only': 1.5806994438171387, 'fused_vs_marginal_gap': 0.9035254716873169, 'gap_related': 0.9035254716873169, 'gap_unrelated': nan}
    12100 {'loss': 1.275348424911499, 'nll_fused_A': 0.6175478100776672, 'nll_fused_BinA': 0.6158326864242554, 'nll_marg_A': 1.5808382034301758, 'nll_marg_BinA_only': 1.580098032951355, 'fused_vs_marginal_gap': 0.9642653465270996, 'gap_related': 0.9642653465270996, 'gap_unrelated': nan}
    12200 {'loss': 1.3612234592437744, 'nll_fused_A': 0.6606955528259277, 'nll_fused_BinA': 0.6698543429374695, 'nll_marg_A': 1.64386785030365, 'nll_marg_BinA_only': 1.6400425434112549, 'fused_vs_marginal_gap': 0.970188319683075, 'gap_related': 0.970188319683075, 'gap_unrelated': nan}
    12300 {'loss': 1.3543404340744019, 'nll_fused_A': 0.674285888671875, 'nll_fused_BinA': 0.6635485887527466, 'nll_marg_A': 1.6283535957336426, 'nll_marg_BinA_only': 1.6299381256103516, 'fused_vs_marginal_gap': 0.966389536857605, 'gap_related': 0.9663894772529602, 'gap_unrelated': nan}
    12400 {'loss': 1.3277106285095215, 'nll_fused_A': 0.6390053629875183, 'nll_fused_BinA': 0.652397871017456, 'nll_marg_A': 1.6120370626449585, 'nll_marg_BinA_only': 1.6282966136932373, 'fused_vs_marginal_gap': 0.9758986234664917, 'gap_related': 0.9758986234664917, 'gap_unrelated': nan}
    12500 {'loss': 1.3656220436096191, 'nll_fused_A': 0.678584098815918, 'nll_fused_BinA': 0.6748306751251221, 'nll_marg_A': 1.624053716659546, 'nll_marg_BinA_only': 1.6284689903259277, 'fused_vs_marginal_gap': 0.9536383152008057, 'gap_related': 0.9536383152008057, 'gap_unrelated': nan}
    12600 {'loss': 1.2879860401153564, 'nll_fused_A': 0.6329398155212402, 'nll_fused_BinA': 0.6260985732078552, 'nll_marg_A': 1.5733519792556763, 'nll_marg_BinA_only': 1.581829309463501, 'fused_vs_marginal_gap': 0.9557307958602905, 'gap_related': 0.9557307958602905, 'gap_unrelated': nan}
    12700 {'loss': 1.365664005279541, 'nll_fused_A': 0.6752214431762695, 'nll_fused_BinA': 0.681519091129303, 'nll_marg_A': 1.6052619218826294, 'nll_marg_BinA_only': 1.6196942329406738, 'fused_vs_marginal_gap': 0.9381752610206604, 'gap_related': 0.9381752014160156, 'gap_unrelated': nan}
    12800 {'loss': 1.343970537185669, 'nll_fused_A': 0.6562820076942444, 'nll_fused_BinA': 0.6613427400588989, 'nll_marg_A': 1.619144082069397, 'nll_marg_BinA_only': 1.6113475561141968, 'fused_vs_marginal_gap': 0.9500048160552979, 'gap_related': 0.9500048160552979, 'gap_unrelated': nan}
    12900 {'loss': 1.2635784149169922, 'nll_fused_A': 0.6003127098083496, 'nll_fused_BinA': 0.606329619884491, 'nll_marg_A': 1.5905163288116455, 'nll_marg_BinA_only': 1.5924460887908936, 'fused_vs_marginal_gap': 0.986116349697113, 'gap_related': 0.9861164093017578, 'gap_unrelated': nan}
    13000 {'loss': 1.3158087730407715, 'nll_fused_A': 0.6462337374687195, 'nll_fused_BinA': 0.6444686055183411, 'nll_marg_A': 1.5915666818618774, 'nll_marg_BinA_only': 1.5817431211471558, 'fused_vs_marginal_gap': 0.9372745752334595, 'gap_related': 0.9372745156288147, 'gap_unrelated': nan}
    13100 {'loss': 1.3178242444992065, 'nll_fused_A': 0.6357064247131348, 'nll_fused_BinA': 0.6359021067619324, 'nll_marg_A': 1.6373672485351562, 'nll_marg_BinA_only': 1.6305166482925415, 'fused_vs_marginal_gap': 0.9946145415306091, 'gap_related': 0.9946144819259644, 'gap_unrelated': nan}
    13200 {'loss': 1.3255375623703003, 'nll_fused_A': 0.6400735378265381, 'nll_fused_BinA': 0.6432579755783081, 'nll_marg_A': 1.634191632270813, 'nll_marg_BinA_only': 1.6367772817611694, 'fused_vs_marginal_gap': 0.9935193061828613, 'gap_related': 0.9935193061828613, 'gap_unrelated': nan}
    13300 {'loss': 1.336843729019165, 'nll_fused_A': 0.6586720943450928, 'nll_fused_BinA': 0.6616220474243164, 'nll_marg_A': 1.592067003250122, 'nll_marg_BinA_only': 1.594651699066162, 'fused_vs_marginal_gap': 0.9330297112464905, 'gap_related': 0.9330296516418457, 'gap_unrelated': nan}
    13400 {'loss': 1.2834471464157104, 'nll_fused_A': 0.6294580101966858, 'nll_fused_BinA': 0.6157284379005432, 'nll_marg_A': 1.5962709188461304, 'nll_marg_BinA_only': 1.590552806854248, 'fused_vs_marginal_gap': 0.9748244285583496, 'gap_related': 0.9748244285583496, 'gap_unrelated': nan}
    13500 {'loss': 1.3070032596588135, 'nll_fused_A': 0.6382272243499756, 'nll_fused_BinA': 0.6443980932235718, 'nll_marg_A': 1.5704565048217773, 'nll_marg_BinA_only': 1.5686428546905518, 'fused_vs_marginal_gap': 0.92424476146698, 'gap_related': 0.9242448210716248, 'gap_unrelated': nan}
    13600 {'loss': 1.3377306461334229, 'nll_fused_A': 0.6635479927062988, 'nll_fused_BinA': 0.653469979763031, 'nll_marg_A': 1.6173210144042969, 'nll_marg_BinA_only': 1.6092097759246826, 'fused_vs_marginal_gap': 0.9557398557662964, 'gap_related': 0.9557397961616516, 'gap_unrelated': nan}
    13700 {'loss': 1.2891740798950195, 'nll_fused_A': 0.6313537955284119, 'nll_fused_BinA': 0.6312369108200073, 'nll_marg_A': 1.5617703199386597, 'nll_marg_BinA_only': 1.567615032196045, 'fused_vs_marginal_gap': 0.9363780617713928, 'gap_related': 0.9363780617713928, 'gap_unrelated': nan}
    13800 {'loss': 1.230851173400879, 'nll_fused_A': 0.582879364490509, 'nll_fused_BinA': 0.5855897665023804, 'nll_marg_A': 1.5679922103881836, 'nll_marg_BinA_only': 1.5723686218261719, 'fused_vs_marginal_gap': 0.9867788553237915, 'gap_related': 0.9867788553237915, 'gap_unrelated': nan}
    13900 {'loss': 1.3084604740142822, 'nll_fused_A': 0.6453920006752014, 'nll_fused_BinA': 0.6374574303627014, 'nll_marg_A': 1.5912845134735107, 'nll_marg_BinA_only': 1.5776331424713135, 'fused_vs_marginal_gap': 0.9401757717132568, 'gap_related': 0.9401757717132568, 'gap_unrelated': nan}
    14000 {'loss': 1.2834900617599487, 'nll_fused_A': 0.6302982568740845, 'nll_fused_BinA': 0.6160025000572205, 'nll_marg_A': 1.5946602821350098, 'nll_marg_BinA_only': 1.603667974472046, 'fused_vs_marginal_gap': 0.9876655340194702, 'gap_related': 0.9876654744148254, 'gap_unrelated': nan}
    14100 {'loss': 1.2922675609588623, 'nll_fused_A': 0.6321198344230652, 'nll_fused_BinA': 0.6238885521888733, 'nll_marg_A': 1.5958099365234375, 'nll_marg_BinA_only': 1.5886691808700562, 'fused_vs_marginal_gap': 0.9647806286811829, 'gap_related': 0.9647805690765381, 'gap_unrelated': nan}
    14200 {'loss': 1.2468361854553223, 'nll_fused_A': 0.5908763408660889, 'nll_fused_BinA': 0.5930927395820618, 'nll_marg_A': 1.5882682800292969, 'nll_marg_BinA_only': 1.5882508754730225, 'fused_vs_marginal_gap': 0.9951581358909607, 'gap_related': 0.9951580762863159, 'gap_unrelated': nan}
    14300 {'loss': 1.1977958679199219, 'nll_fused_A': 0.5530436635017395, 'nll_fused_BinA': 0.5525665879249573, 'nll_marg_A': 1.5977206230163574, 'nll_marg_BinA_only': 1.5999062061309814, 'fused_vs_marginal_gap': 1.0473395586013794, 'gap_related': 1.0473395586013794, 'gap_unrelated': nan}
    14400 {'loss': 1.2487058639526367, 'nll_fused_A': 0.5783395767211914, 'nll_fused_BinA': 0.6020594835281372, 'nll_marg_A': 1.577148199081421, 'nll_marg_BinA_only': 1.5870585441589355, 'fused_vs_marginal_gap': 0.9849990606307983, 'gap_related': 0.9849991202354431, 'gap_unrelated': nan}
    14500 {'loss': 1.2154676914215088, 'nll_fused_A': 0.5607357621192932, 'nll_fused_BinA': 0.5783525705337524, 'nll_marg_A': 1.562981128692627, 'nll_marg_BinA_only': 1.5668540000915527, 'fused_vs_marginal_gap': 0.9885013103485107, 'gap_related': 0.9885013699531555, 'gap_unrelated': nan}
    14600 {'loss': 1.1683945655822754, 'nll_fused_A': 0.5400193929672241, 'nll_fused_BinA': 0.5414664149284363, 'nll_marg_A': 1.5497410297393799, 'nll_marg_BinA_only': 1.5580592155456543, 'fused_vs_marginal_gap': 1.0165927410125732, 'gap_related': 1.0165927410125732, 'gap_unrelated': nan}
    14700 {'loss': 1.2490110397338867, 'nll_fused_A': 0.5827982425689697, 'nll_fused_BinA': 0.5865412950515747, 'nll_marg_A': 1.6254342794418335, 'nll_marg_BinA_only': 1.6004250049591064, 'fused_vs_marginal_gap': 1.0138835906982422, 'gap_related': 1.0138837099075317, 'gap_unrelated': nan}
    14800 {'loss': 1.2198773622512817, 'nll_fused_A': 0.5876219272613525, 'nll_fused_BinA': 0.5766323804855347, 'nll_marg_A': 1.5565277338027954, 'nll_marg_BinA_only': 1.5485063791275024, 'fused_vs_marginal_gap': 0.9718738794326782, 'gap_related': 0.971873939037323, 'gap_unrelated': nan}
    14900 {'loss': 1.2159541845321655, 'nll_fused_A': 0.5656865835189819, 'nll_fused_BinA': 0.5799907445907593, 'nll_marg_A': 1.5541914701461792, 'nll_marg_BinA_only': 1.559528112411499, 'fused_vs_marginal_gap': 0.9795374274253845, 'gap_related': 0.9795373678207397, 'gap_unrelated': nan}
    15000 {'loss': 1.2362037897109985, 'nll_fused_A': 0.5988754034042358, 'nll_fused_BinA': 0.5918035507202148, 'nll_marg_A': 1.54912531375885, 'nll_marg_BinA_only': 1.5471136569976807, 'fused_vs_marginal_gap': 0.955310046672821, 'gap_related': 0.9553099870681763, 'gap_unrelated': nan}
    15100 {'loss': 1.2155128717422485, 'nll_fused_A': 0.5568649768829346, 'nll_fused_BinA': 0.5774869918823242, 'nll_marg_A': 1.5698877573013306, 'nll_marg_BinA_only': 1.552598476409912, 'fused_vs_marginal_gap': 0.9751114845275879, 'gap_related': 0.9751114845275879, 'gap_unrelated': nan}
    15200 {'loss': 1.250961184501648, 'nll_fused_A': 0.6005071401596069, 'nll_fused_BinA': 0.6007078886032104, 'nll_marg_A': 1.567003846168518, 'nll_marg_BinA_only': 1.5726183652877808, 'fused_vs_marginal_gap': 0.9719104170799255, 'gap_related': 0.9719104766845703, 'gap_unrelated': nan}
    15300 {'loss': 1.2056171894073486, 'nll_fused_A': 0.5668007731437683, 'nll_fused_BinA': 0.564349889755249, 'nll_marg_A': 1.5707566738128662, 'nll_marg_BinA_only': 1.5687334537506104, 'fused_vs_marginal_gap': 1.0043835639953613, 'gap_related': 1.0043835639953613, 'gap_unrelated': nan}
    15400 {'loss': 1.2111248970031738, 'nll_fused_A': 0.5624347925186157, 'nll_fused_BinA': 0.57515549659729, 'nll_marg_A': 1.5574629306793213, 'nll_marg_BinA_only': 1.574996829032898, 'fused_vs_marginal_gap': 0.9998413324356079, 'gap_related': 0.9998413324356079, 'gap_unrelated': nan}
    15500 {'loss': 1.1503493785858154, 'nll_fused_A': 0.5278428792953491, 'nll_fused_BinA': 0.5249229669570923, 'nll_marg_A': 1.556911826133728, 'nll_marg_BinA_only': 1.583612322807312, 'fused_vs_marginal_gap': 1.0586892366409302, 'gap_related': 1.0586893558502197, 'gap_unrelated': nan}
    15600 {'loss': 1.1994729042053223, 'nll_fused_A': 0.5548272132873535, 'nll_fused_BinA': 0.5634450912475586, 'nll_marg_A': 1.5652657747268677, 'nll_marg_BinA_only': 1.577796220779419, 'fused_vs_marginal_gap': 1.0143508911132812, 'gap_related': 1.0143508911132812, 'gap_unrelated': nan}
    15700 {'loss': 1.2323877811431885, 'nll_fused_A': 0.5767168402671814, 'nll_fused_BinA': 0.5856051445007324, 'nll_marg_A': 1.5792251825332642, 'nll_marg_BinA_only': 1.586838722229004, 'fused_vs_marginal_gap': 1.0012335777282715, 'gap_related': 1.001233696937561, 'gap_unrelated': nan}
    15800 {'loss': 1.1744146347045898, 'nll_fused_A': 0.5463817119598389, 'nll_fused_BinA': 0.5429584980010986, 'nll_marg_A': 1.5584721565246582, 'nll_marg_BinA_only': 1.5517433881759644, 'fused_vs_marginal_gap': 1.0087848901748657, 'gap_related': 1.0087848901748657, 'gap_unrelated': nan}
    15900 {'loss': 1.2445688247680664, 'nll_fused_A': 0.6097419261932373, 'nll_fused_BinA': 0.5960962176322937, 'nll_marg_A': 1.5518336296081543, 'nll_marg_BinA_only': 1.558295726776123, 'fused_vs_marginal_gap': 0.9621994495391846, 'gap_related': 0.9621995687484741, 'gap_unrelated': nan}
    16000 {'loss': 1.164749264717102, 'nll_fused_A': 0.5446255803108215, 'nll_fused_BinA': 0.5265954732894897, 'nll_marg_A': 1.5825536251068115, 'nll_marg_BinA_only': 1.5699161291122437, 'fused_vs_marginal_gap': 1.043320655822754, 'gap_related': 1.043320655822754, 'gap_unrelated': nan}
    16100 {'loss': 1.1238460540771484, 'nll_fused_A': 0.5076519250869751, 'nll_fused_BinA': 0.5094271302223206, 'nll_marg_A': 1.5404114723205566, 'nll_marg_BinA_only': 1.5261201858520508, 'fused_vs_marginal_gap': 1.016693115234375, 'gap_related': 1.016693115234375, 'gap_unrelated': nan}
    16200 {'loss': 1.1423122882843018, 'nll_fused_A': 0.5244660377502441, 'nll_fused_BinA': 0.5224003791809082, 'nll_marg_A': 1.5419070720672607, 'nll_marg_BinA_only': 1.5405044555664062, 'fused_vs_marginal_gap': 1.018104076385498, 'gap_related': 1.018104076385498, 'gap_unrelated': nan}
    16300 {'loss': 1.1889467239379883, 'nll_fused_A': 0.5506916046142578, 'nll_fused_BinA': 0.5582126975059509, 'nll_marg_A': 1.5517549514770508, 'nll_marg_BinA_only': 1.5528615713119507, 'fused_vs_marginal_gap': 0.994648814201355, 'gap_related': 0.9946487545967102, 'gap_unrelated': nan}
    16400 {'loss': 1.205232858657837, 'nll_fused_A': 0.5715359449386597, 'nll_fused_BinA': 0.5604082345962524, 'nll_marg_A': 1.5778794288635254, 'nll_marg_BinA_only': 1.565909743309021, 'fused_vs_marginal_gap': 1.0055015087127686, 'gap_related': 1.005501389503479, 'gap_unrelated': nan}
    16500 {'loss': 1.1816006898880005, 'nll_fused_A': 0.5351263284683228, 'nll_fused_BinA': 0.5492628216743469, 'nll_marg_A': 1.5726664066314697, 'nll_marg_BinA_only': 1.5777486562728882, 'fused_vs_marginal_gap': 1.0284857749938965, 'gap_related': 1.0284857749938965, 'gap_unrelated': nan}
    16600 {'loss': 1.119154930114746, 'nll_fused_A': 0.49990883469581604, 'nll_fused_BinA': 0.512321949005127, 'nll_marg_A': 1.5228675603866577, 'nll_marg_BinA_only': 1.535792350769043, 'fused_vs_marginal_gap': 1.023470401763916, 'gap_related': 1.023470401763916, 'gap_unrelated': nan}
    16700 {'loss': 1.1786820888519287, 'nll_fused_A': 0.5401422381401062, 'nll_fused_BinA': 0.5373582243919373, 'nll_marg_A': 1.5976039171218872, 'nll_marg_BinA_only': 1.5812556743621826, 'fused_vs_marginal_gap': 1.0438975095748901, 'gap_related': 1.0438973903656006, 'gap_unrelated': nan}
    16800 {'loss': 1.1017615795135498, 'nll_fused_A': 0.48882272839546204, 'nll_fused_BinA': 0.4892539978027344, 'nll_marg_A': 1.552869200706482, 'nll_marg_BinA_only': 1.5317022800445557, 'fused_vs_marginal_gap': 1.0424482822418213, 'gap_related': 1.0424482822418213, 'gap_unrelated': nan}
    16900 {'loss': 1.2391703128814697, 'nll_fused_A': 0.5829799175262451, 'nll_fused_BinA': 0.5844528675079346, 'nll_marg_A': 1.5994117259979248, 'nll_marg_BinA_only': 1.6075727939605713, 'fused_vs_marginal_gap': 1.0231199264526367, 'gap_related': 1.0231200456619263, 'gap_unrelated': nan}
    17000 {'loss': 1.1366691589355469, 'nll_fused_A': 0.507007360458374, 'nll_fused_BinA': 0.5064228177070618, 'nll_marg_A': 1.5938136577606201, 'nll_marg_BinA_only': 1.5836327075958252, 'fused_vs_marginal_gap': 1.0772099494934082, 'gap_related': 1.0772098302841187, 'gap_unrelated': nan}
    17100 {'loss': 1.0674073696136475, 'nll_fused_A': 0.4696173071861267, 'nll_fused_BinA': 0.4763166010379791, 'nll_marg_A': 1.5006850957870483, 'nll_marg_BinA_only': 1.4937782287597656, 'fused_vs_marginal_gap': 1.0174615383148193, 'gap_related': 1.0174615383148193, 'gap_unrelated': nan}
    17200 {'loss': 1.049530029296875, 'nll_fused_A': 0.4408418834209442, 'nll_fused_BinA': 0.4545239210128784, 'nll_marg_A': 1.5425119400024414, 'nll_marg_BinA_only': 1.5389132499694824, 'fused_vs_marginal_gap': 1.0843894481658936, 'gap_related': 1.0843894481658936, 'gap_unrelated': nan}
    17300 {'loss': 1.0929369926452637, 'nll_fused_A': 0.5060254335403442, 'nll_fused_BinA': 0.48225027322769165, 'nll_marg_A': 1.5295970439910889, 'nll_marg_BinA_only': 1.537306547164917, 'fused_vs_marginal_gap': 1.0550563335418701, 'gap_related': 1.0550563335418701, 'gap_unrelated': nan}
    17400 {'loss': 1.1690967082977295, 'nll_fused_A': 0.5316057801246643, 'nll_fused_BinA': 0.5458433628082275, 'nll_marg_A': 1.5459052324295044, 'nll_marg_BinA_only': 1.5532171726226807, 'fused_vs_marginal_gap': 1.0073739290237427, 'gap_related': 1.0073738098144531, 'gap_unrelated': nan}
    17500 {'loss': 1.126863956451416, 'nll_fused_A': 0.5187349319458008, 'nll_fused_BinA': 0.505165696144104, 'nll_marg_A': 1.553592562675476, 'nll_marg_BinA_only': 1.537858009338379, 'fused_vs_marginal_gap': 1.032692313194275, 'gap_related': 1.032692313194275, 'gap_unrelated': nan}
    17600 {'loss': 1.0865633487701416, 'nll_fused_A': 0.4842337667942047, 'nll_fused_BinA': 0.4690348505973816, 'nll_marg_A': 1.574194312095642, 'nll_marg_BinA_only': 1.5663725137710571, 'fused_vs_marginal_gap': 1.0973376035690308, 'gap_related': 1.0973377227783203, 'gap_unrelated': nan}
    17700 {'loss': 1.0906662940979004, 'nll_fused_A': 0.5066165924072266, 'nll_fused_BinA': 0.4827457070350647, 'nll_marg_A': 1.5197855234146118, 'nll_marg_BinA_only': 1.502959966659546, 'fused_vs_marginal_gap': 1.020214319229126, 'gap_related': 1.020214319229126, 'gap_unrelated': nan}
    17800 {'loss': 1.1738967895507812, 'nll_fused_A': 0.5244095325469971, 'nll_fused_BinA': 0.5381863117218018, 'nll_marg_A': 1.5946252346038818, 'nll_marg_BinA_only': 1.5934312343597412, 'fused_vs_marginal_gap': 1.0552449226379395, 'gap_related': 1.055245041847229, 'gap_unrelated': nan}
    17900 {'loss': 1.14493727684021, 'nll_fused_A': 0.5122804641723633, 'nll_fused_BinA': 0.5175997018814087, 'nll_marg_A': 1.5788447856903076, 'nll_marg_BinA_only': 1.572507619857788, 'fused_vs_marginal_gap': 1.0549077987670898, 'gap_related': 1.0549077987670898, 'gap_unrelated': nan}
    18000 {'loss': 1.04789400100708, 'nll_fused_A': 0.44675907492637634, 'nll_fused_BinA': 0.4524644911289215, 'nll_marg_A': 1.5380058288574219, 'nll_marg_BinA_only': 1.5567917823791504, 'fused_vs_marginal_gap': 1.1043272018432617, 'gap_related': 1.1043272018432617, 'gap_unrelated': nan}
    18100 {'loss': 1.046109676361084, 'nll_fused_A': 0.45561063289642334, 'nll_fused_BinA': 0.44075918197631836, 'nll_marg_A': 1.56222403049469, 'nll_marg_BinA_only': 1.5560637712478638, 'fused_vs_marginal_gap': 1.1153045892715454, 'gap_related': 1.1153045892715454, 'gap_unrelated': nan}
    18200 {'loss': 1.1029413938522339, 'nll_fused_A': 0.4993543028831482, 'nll_fused_BinA': 0.48357969522476196, 'nll_marg_A': 1.5651847124099731, 'nll_marg_BinA_only': 1.5559465885162354, 'fused_vs_marginal_gap': 1.0723669528961182, 'gap_related': 1.0723669528961182, 'gap_unrelated': nan}
    18300 {'loss': 1.012448787689209, 'nll_fused_A': 0.43253597617149353, 'nll_fused_BinA': 0.42831355333328247, 'nll_marg_A': 1.5145814418792725, 'nll_marg_BinA_only': 1.5091800689697266, 'fused_vs_marginal_gap': 1.0808663368225098, 'gap_related': 1.0808663368225098, 'gap_unrelated': nan}
    18400 {'loss': 1.1081433296203613, 'nll_fused_A': 0.511965274810791, 'nll_fused_BinA': 0.5070071220397949, 'nll_marg_A': 1.4918217658996582, 'nll_marg_BinA_only': 1.4912304878234863, 'fused_vs_marginal_gap': 0.9842232465744019, 'gap_related': 0.9842233657836914, 'gap_unrelated': nan}
    18500 {'loss': 0.9909054040908813, 'nll_fused_A': 0.4186420440673828, 'nll_fused_BinA': 0.41761329770088196, 'nll_marg_A': 1.4923313856124878, 'nll_marg_BinA_only': 1.4993922710418701, 'fused_vs_marginal_gap': 1.081778883934021, 'gap_related': 1.081778883934021, 'gap_unrelated': nan}
    18600 {'loss': 1.0296053886413574, 'nll_fused_A': 0.4519484341144562, 'nll_fused_BinA': 0.4429687261581421, 'nll_marg_A': 1.5035068988800049, 'nll_marg_BinA_only': 1.4966658353805542, 'fused_vs_marginal_gap': 1.053697109222412, 'gap_related': 1.053697109222412, 'gap_unrelated': nan}
    18700 {'loss': 1.1429697275161743, 'nll_fused_A': 0.5236712694168091, 'nll_fused_BinA': 0.5171485543251038, 'nll_marg_A': 1.56239914894104, 'nll_marg_BinA_only': 1.5566117763519287, 'fused_vs_marginal_gap': 1.0394631624221802, 'gap_related': 1.0394631624221802, 'gap_unrelated': nan}
    18800 {'loss': 1.1088342666625977, 'nll_fused_A': 0.4951007068157196, 'nll_fused_BinA': 0.49469804763793945, 'nll_marg_A': 1.5520200729370117, 'nll_marg_BinA_only': 1.5550870895385742, 'fused_vs_marginal_gap': 1.0603890419006348, 'gap_related': 1.0603890419006348, 'gap_unrelated': nan}
    18900 {'loss': 1.056062936782837, 'nll_fused_A': 0.4642409682273865, 'nll_fused_BinA': 0.46712467074394226, 'nll_marg_A': 1.4988865852355957, 'nll_marg_BinA_only': 1.5064306259155273, 'fused_vs_marginal_gap': 1.0393059253692627, 'gap_related': 1.0393060445785522, 'gap_unrelated': nan}
    19000 {'loss': 1.012638807296753, 'nll_fused_A': 0.43013691902160645, 'nll_fused_BinA': 0.4265727698802948, 'nll_marg_A': 1.52341628074646, 'nll_marg_BinA_only': 1.525413155555725, 'fused_vs_marginal_gap': 1.0988404750823975, 'gap_related': 1.0988404750823975, 'gap_unrelated': nan}
    19100 {'loss': 1.0889915227890015, 'nll_fused_A': 0.493297278881073, 'nll_fused_BinA': 0.4859886169433594, 'nll_marg_A': 1.5167121887207031, 'nll_marg_BinA_only': 1.5101439952850342, 'fused_vs_marginal_gap': 1.0241553783416748, 'gap_related': 1.0241554975509644, 'gap_unrelated': nan}
    19200 {'loss': 1.0915082693099976, 'nll_fused_A': 0.5000516176223755, 'nll_fused_BinA': 0.485890656709671, 'nll_marg_A': 1.5186738967895508, 'nll_marg_BinA_only': 1.5227020978927612, 'fused_vs_marginal_gap': 1.036811351776123, 'gap_related': 1.036811351776123, 'gap_unrelated': nan}
    19300 {'loss': 1.0779006481170654, 'nll_fused_A': 0.4545128345489502, 'nll_fused_BinA': 0.4777818024158478, 'nll_marg_A': 1.5458831787109375, 'nll_marg_BinA_only': 1.5593676567077637, 'fused_vs_marginal_gap': 1.0815858840942383, 'gap_related': 1.0815858840942383, 'gap_unrelated': nan}
    19400 {'loss': 1.0891180038452148, 'nll_fused_A': 0.4777498245239258, 'nll_fused_BinA': 0.4804452061653137, 'nll_marg_A': 1.551159381866455, 'nll_marg_BinA_only': 1.56049382686615, 'fused_vs_marginal_gap': 1.0800485610961914, 'gap_related': 1.0800485610961914, 'gap_unrelated': nan}
    19500 {'loss': 1.0406991243362427, 'nll_fused_A': 0.4386921525001526, 'nll_fused_BinA': 0.4364815354347229, 'nll_marg_A': 1.5753663778305054, 'nll_marg_BinA_only': 1.5677876472473145, 'fused_vs_marginal_gap': 1.1313060522079468, 'gap_related': 1.1313060522079468, 'gap_unrelated': nan}
    19600 {'loss': 1.1332736015319824, 'nll_fused_A': 0.5108344554901123, 'nll_fused_BinA': 0.503348171710968, 'nll_marg_A': 1.5889170169830322, 'nll_marg_BinA_only': 1.5897581577301025, 'fused_vs_marginal_gap': 1.0864100456237793, 'gap_related': 1.0864100456237793, 'gap_unrelated': nan}
    19700 {'loss': 0.9626330137252808, 'nll_fused_A': 0.39108848571777344, 'nll_fused_BinA': 0.38985395431518555, 'nll_marg_A': 1.5181748867034912, 'nll_marg_BinA_only': 1.5229703187942505, 'fused_vs_marginal_gap': 1.133116364479065, 'gap_related': 1.1331162452697754, 'gap_unrelated': nan}
    19800 {'loss': 1.0294567346572876, 'nll_fused_A': 0.4323952794075012, 'nll_fused_BinA': 0.44322168827056885, 'nll_marg_A': 1.521721601486206, 'nll_marg_BinA_only': 1.5345536470413208, 'fused_vs_marginal_gap': 1.091331958770752, 'gap_related': 1.091331958770752, 'gap_unrelated': nan}
    19900 {'loss': 0.9943684339523315, 'nll_fused_A': 0.4203500747680664, 'nll_fused_BinA': 0.415629506111145, 'nll_marg_A': 1.508779525756836, 'nll_marg_BinA_only': 1.5153957605361938, 'fused_vs_marginal_gap': 1.0997662544250488, 'gap_related': 1.0997662544250488, 'gap_unrelated': nan}
    20000 {'loss': 1.027177095413208, 'nll_fused_A': 0.4384118914604187, 'nll_fused_BinA': 0.4386392831802368, 'nll_marg_A': 1.5233806371688843, 'nll_marg_BinA_only': 1.5188521146774292, 'fused_vs_marginal_gap': 1.0802128314971924, 'gap_related': 1.0802128314971924, 'gap_unrelated': nan}
    20100 {'loss': 1.0329591035842896, 'nll_fused_A': 0.4399568438529968, 'nll_fused_BinA': 0.4484415054321289, 'nll_marg_A': 1.5084350109100342, 'nll_marg_BinA_only': 1.5315866470336914, 'fused_vs_marginal_gap': 1.0831451416015625, 'gap_related': 1.0831451416015625, 'gap_unrelated': nan}
    20200 {'loss': 0.8554612398147583, 'nll_fused_A': 0.32052281498908997, 'nll_fused_BinA': 0.3113821744918823, 'nll_marg_A': 1.4930739402770996, 'nll_marg_BinA_only': 1.4881608486175537, 'fused_vs_marginal_gap': 1.1767785549163818, 'gap_related': 1.1767785549163818, 'gap_unrelated': nan}
    20300 {'loss': 0.9841705560684204, 'nll_fused_A': 0.4117357134819031, 'nll_fused_BinA': 0.41432082653045654, 'nll_marg_A': 1.4877631664276123, 'nll_marg_BinA_only': 1.4908307790756226, 'fused_vs_marginal_gap': 1.076509952545166, 'gap_related': 1.076509952545166, 'gap_unrelated': nan}
    20400 {'loss': 1.0681877136230469, 'nll_fused_A': 0.45734483003616333, 'nll_fused_BinA': 0.4617684483528137, 'nll_marg_A': 1.5640524625778198, 'nll_marg_BinA_only': 1.5528658628463745, 'fused_vs_marginal_gap': 1.091097354888916, 'gap_related': 1.0910974740982056, 'gap_unrelated': nan}
    20500 {'loss': 0.8901738524436951, 'nll_fused_A': 0.3561916947364807, 'nll_fused_BinA': 0.3448609709739685, 'nll_marg_A': 1.4615178108215332, 'nll_marg_BinA_only': 1.4769597053527832, 'fused_vs_marginal_gap': 1.1320987939834595, 'gap_related': 1.1320987939834595, 'gap_unrelated': nan}
    20600 {'loss': 1.0264654159545898, 'nll_fused_A': 0.4287300407886505, 'nll_fused_BinA': 0.4471140503883362, 'nll_marg_A': 1.50244140625, 'nll_marg_BinA_only': 1.5314489603042603, 'fused_vs_marginal_gap': 1.0843349695205688, 'gap_related': 1.0843348503112793, 'gap_unrelated': nan}
    20700 {'loss': 0.8909765481948853, 'nll_fused_A': 0.34639501571655273, 'nll_fused_BinA': 0.34611284732818604, 'nll_marg_A': 1.4698172807693481, 'nll_marg_BinA_only': 1.4831968545913696, 'fused_vs_marginal_gap': 1.137083888053894, 'gap_related': 1.1370840072631836, 'gap_unrelated': nan}
    20800 {'loss': 1.0142385959625244, 'nll_fused_A': 0.40745648741722107, 'nll_fused_BinA': 0.41778215765953064, 'nll_marg_A': 1.5807313919067383, 'nll_marg_BinA_only': 1.5974400043487549, 'fused_vs_marginal_gap': 1.1796579360961914, 'gap_related': 1.1796579360961914, 'gap_unrelated': nan}
    20900 {'loss': 0.9757459163665771, 'nll_fused_A': 0.4059215784072876, 'nll_fused_BinA': 0.39168664813041687, 'nll_marg_A': 1.5409425497055054, 'nll_marg_BinA_only': 1.5217933654785156, 'fused_vs_marginal_gap': 1.1301066875457764, 'gap_related': 1.1301066875457764, 'gap_unrelated': nan}
    21000 {'loss': 1.0782876014709473, 'nll_fused_A': 0.4609593152999878, 'nll_fused_BinA': 0.4783148765563965, 'nll_marg_A': 1.538949728012085, 'nll_marg_BinA_only': 1.5586388111114502, 'fused_vs_marginal_gap': 1.0803239345550537, 'gap_related': 1.0803239345550537, 'gap_unrelated': nan}
    21100 {'loss': 0.9331802725791931, 'nll_fused_A': 0.37821274995803833, 'nll_fused_BinA': 0.367658793926239, 'nll_marg_A': 1.5068588256835938, 'nll_marg_BinA_only': 1.5054322481155396, 'fused_vs_marginal_gap': 1.1377735137939453, 'gap_related': 1.1377735137939453, 'gap_unrelated': nan}
    21200 {'loss': 0.9258478879928589, 'nll_fused_A': 0.35428041219711304, 'nll_fused_BinA': 0.36843833327293396, 'nll_marg_A': 1.503751277923584, 'nll_marg_BinA_only': 1.5048255920410156, 'fused_vs_marginal_gap': 1.1363873481750488, 'gap_related': 1.1363872289657593, 'gap_unrelated': nan}
    21300 {'loss': 0.8362447023391724, 'nll_fused_A': 0.304459810256958, 'nll_fused_BinA': 0.30628326535224915, 'nll_marg_A': 1.462078332901001, 'nll_marg_BinA_only': 1.4711298942565918, 'fused_vs_marginal_gap': 1.1648465394973755, 'gap_related': 1.164846658706665, 'gap_unrelated': nan}
    21400 {'loss': 0.9575715661048889, 'nll_fused_A': 0.3782854676246643, 'nll_fused_BinA': 0.3880993127822876, 'nll_marg_A': 1.5199553966522217, 'nll_marg_BinA_only': 1.5343213081359863, 'fused_vs_marginal_gap': 1.1462221145629883, 'gap_related': 1.1462221145629883, 'gap_unrelated': nan}
    21500 {'loss': 0.8104739785194397, 'nll_fused_A': 0.3083454668521881, 'nll_fused_BinA': 0.291944682598114, 'nll_marg_A': 1.4200854301452637, 'nll_marg_BinA_only': 1.4276918172836304, 'fused_vs_marginal_gap': 1.1357471942901611, 'gap_related': 1.1357473134994507, 'gap_unrelated': nan}
    21600 {'loss': 1.0422608852386475, 'nll_fused_A': 0.43193796277046204, 'nll_fused_BinA': 0.4399873614311218, 'nll_marg_A': 1.575640082359314, 'nll_marg_BinA_only': 1.576145887374878, 'fused_vs_marginal_gap': 1.1361584663391113, 'gap_related': 1.1361584663391113, 'gap_unrelated': nan}
    21700 {'loss': 1.0180087089538574, 'nll_fused_A': 0.42422354221343994, 'nll_fused_BinA': 0.41770780086517334, 'nll_marg_A': 1.5767793655395508, 'nll_marg_BinA_only': 1.5682663917541504, 'fused_vs_marginal_gap': 1.150558590888977, 'gap_related': 1.150558590888977, 'gap_unrelated': nan}
    21800 {'loss': 0.9933284521102905, 'nll_fused_A': 0.4163718819618225, 'nll_fused_BinA': 0.4132707118988037, 'nll_marg_A': 1.5171539783477783, 'nll_marg_BinA_only': 1.5353572368621826, 'fused_vs_marginal_gap': 1.122086524963379, 'gap_related': 1.122086524963379, 'gap_unrelated': nan}
    21900 {'loss': 0.9495381116867065, 'nll_fused_A': 0.37476927042007446, 'nll_fused_BinA': 0.38275647163391113, 'nll_marg_A': 1.5145026445388794, 'nll_marg_BinA_only': 1.5213549137115479, 'fused_vs_marginal_gap': 1.1385984420776367, 'gap_related': 1.1385984420776367, 'gap_unrelated': nan}
    22000 {'loss': 0.8283774852752686, 'nll_fused_A': 0.29737943410873413, 'nll_fused_BinA': 0.2992596924304962, 'nll_marg_A': 1.466346263885498, 'nll_marg_BinA_only': 1.467787265777588, 'fused_vs_marginal_gap': 1.1685274839401245, 'gap_related': 1.168527603149414, 'gap_unrelated': nan}
    22100 {'loss': 0.9610933661460876, 'nll_fused_A': 0.3802143931388855, 'nll_fused_BinA': 0.3887330889701843, 'nll_marg_A': 1.5276529788970947, 'nll_marg_BinA_only': 1.547053337097168, 'fused_vs_marginal_gap': 1.1583203077316284, 'gap_related': 1.1583201885223389, 'gap_unrelated': nan}
    22200 {'loss': 0.9167689681053162, 'nll_fused_A': 0.3527716398239136, 'nll_fused_BinA': 0.3583894371986389, 'nll_marg_A': 1.5084933042526245, 'nll_marg_BinA_only': 1.516390323638916, 'fused_vs_marginal_gap': 1.1580009460449219, 'gap_related': 1.1580009460449219, 'gap_unrelated': nan}
    22300 {'loss': 0.952843427658081, 'nll_fused_A': 0.37988102436065674, 'nll_fused_BinA': 0.38284364342689514, 'nll_marg_A': 1.520117998123169, 'nll_marg_BinA_only': 1.5486364364624023, 'fused_vs_marginal_gap': 1.1657929420471191, 'gap_related': 1.1657929420471191, 'gap_unrelated': nan}
    22400 {'loss': 0.8755849599838257, 'nll_fused_A': 0.3198600709438324, 'nll_fused_BinA': 0.3169320523738861, 'nll_marg_A': 1.5423160791397095, 'nll_marg_BinA_only': 1.5279589891433716, 'fused_vs_marginal_gap': 1.211026906967163, 'gap_related': 1.211026906967163, 'gap_unrelated': nan}
    22500 {'loss': 0.8976441621780396, 'nll_fused_A': 0.33656546473503113, 'nll_fused_BinA': 0.3480939269065857, 'nll_marg_A': 1.4952685832977295, 'nll_marg_BinA_only': 1.4948502779006958, 'fused_vs_marginal_gap': 1.1467564105987549, 'gap_related': 1.1467564105987549, 'gap_unrelated': nan}
    22600 {'loss': 0.899813175201416, 'nll_fused_A': 0.33984431624412537, 'nll_fused_BinA': 0.3387090265750885, 'nll_marg_A': 1.5305025577545166, 'nll_marg_BinA_only': 1.5211169719696045, 'fused_vs_marginal_gap': 1.1824078559875488, 'gap_related': 1.1824078559875488, 'gap_unrelated': nan}
    22700 {'loss': 1.0089691877365112, 'nll_fused_A': 0.4219125509262085, 'nll_fused_BinA': 0.41285842657089233, 'nll_marg_A': 1.5651233196258545, 'nll_marg_BinA_only': 1.5795788764953613, 'fused_vs_marginal_gap': 1.1667203903198242, 'gap_related': 1.1667203903198242, 'gap_unrelated': nan}
    22800 {'loss': 0.930309534072876, 'nll_fused_A': 0.34872448444366455, 'nll_fused_BinA': 0.37406668066978455, 'nll_marg_A': 1.505418300628662, 'nll_marg_BinA_only': 1.5120770931243896, 'fused_vs_marginal_gap': 1.1380105018615723, 'gap_related': 1.1380105018615723, 'gap_unrelated': nan}
    22900 {'loss': 0.8540629744529724, 'nll_fused_A': 0.3084346055984497, 'nll_fused_BinA': 0.30929118394851685, 'nll_marg_A': 1.5074713230133057, 'nll_marg_BinA_only': 1.5084155797958374, 'fused_vs_marginal_gap': 1.1991243362426758, 'gap_related': 1.1991243362426758, 'gap_unrelated': nan}
    23000 {'loss': 0.8917762041091919, 'nll_fused_A': 0.35132545232772827, 'nll_fused_BinA': 0.33840757608413696, 'nll_marg_A': 1.4932365417480469, 'nll_marg_BinA_only': 1.4902023077011108, 'fused_vs_marginal_gap': 1.151794672012329, 'gap_related': 1.151794672012329, 'gap_unrelated': nan}
    23100 {'loss': 0.8824911713600159, 'nll_fused_A': 0.33689865469932556, 'nll_fused_BinA': 0.3231816291809082, 'nll_marg_A': 1.5274664163589478, 'nll_marg_BinA_only': 1.5305038690567017, 'fused_vs_marginal_gap': 1.2073222398757935, 'gap_related': 1.2073222398757935, 'gap_unrelated': nan}
    23200 {'loss': 0.9109582304954529, 'nll_fused_A': 0.32340118288993835, 'nll_fused_BinA': 0.3550226092338562, 'nll_marg_A': 1.5297174453735352, 'nll_marg_BinA_only': 1.5367062091827393, 'fused_vs_marginal_gap': 1.1816836595535278, 'gap_related': 1.1816836595535278, 'gap_unrelated': nan}
    23300 {'loss': 0.8924588561058044, 'nll_fused_A': 0.3308682441711426, 'nll_fused_BinA': 0.34137415885925293, 'nll_marg_A': 1.5060806274414062, 'nll_marg_BinA_only': 1.5042898654937744, 'fused_vs_marginal_gap': 1.1629157066345215, 'gap_related': 1.1629157066345215, 'gap_unrelated': nan}
    23400 {'loss': 0.873029351234436, 'nll_fused_A': 0.31504470109939575, 'nll_fused_BinA': 0.3320273458957672, 'nll_marg_A': 1.4882951974868774, 'nll_marg_BinA_only': 1.4943466186523438, 'fused_vs_marginal_gap': 1.1623191833496094, 'gap_related': 1.1623191833496094, 'gap_unrelated': nan}
    23500 {'loss': 0.9075189828872681, 'nll_fused_A': 0.34890586137771606, 'nll_fused_BinA': 0.3415667712688446, 'nll_marg_A': 1.5376014709472656, 'nll_marg_BinA_only': 1.5413695573806763, 'fused_vs_marginal_gap': 1.1998027563095093, 'gap_related': 1.1998027563095093, 'gap_unrelated': nan}
    23600 {'loss': 0.9549000263214111, 'nll_fused_A': 0.37241965532302856, 'nll_fused_BinA': 0.3762610852718353, 'nll_marg_A': 1.556376576423645, 'nll_marg_BinA_only': 1.5506978034973145, 'fused_vs_marginal_gap': 1.1744368076324463, 'gap_related': 1.1744368076324463, 'gap_unrelated': nan}
    23700 {'loss': 0.9051405191421509, 'nll_fused_A': 0.33259475231170654, 'nll_fused_BinA': 0.34363076090812683, 'nll_marg_A': 1.5391044616699219, 'nll_marg_BinA_only': 1.5541565418243408, 'fused_vs_marginal_gap': 1.210525631904602, 'gap_related': 1.210525631904602, 'gap_unrelated': nan}
    23800 {'loss': 0.8702322244644165, 'nll_fused_A': 0.35471096634864807, 'nll_fused_BinA': 0.32753846049308777, 'nll_marg_A': 1.454268217086792, 'nll_marg_BinA_only': 1.4443202018737793, 'fused_vs_marginal_gap': 1.1167818307876587, 'gap_related': 1.1167818307876587, 'gap_unrelated': nan}
    23900 {'loss': 0.9356808066368103, 'nll_fused_A': 0.3429054915904999, 'nll_fused_BinA': 0.3550821542739868, 'nll_marg_A': 1.5924232006072998, 'nll_marg_BinA_only': 1.5761189460754395, 'fused_vs_marginal_gap': 1.221036672592163, 'gap_related': 1.221036672592163, 'gap_unrelated': nan}
    24000 {'loss': 0.864183783531189, 'nll_fused_A': 0.3111279010772705, 'nll_fused_BinA': 0.30647221207618713, 'nll_marg_A': 1.5479106903076172, 'nll_marg_BinA_only': 1.5534396171569824, 'fused_vs_marginal_gap': 1.2469675540924072, 'gap_related': 1.2469675540924072, 'gap_unrelated': nan}
    24100 {'loss': 0.8968640565872192, 'nll_fused_A': 0.3266175389289856, 'nll_fused_BinA': 0.3451758623123169, 'nll_marg_A': 1.512343168258667, 'nll_marg_BinA_only': 1.512147068977356, 'fused_vs_marginal_gap': 1.166971206665039, 'gap_related': 1.166971206665039, 'gap_unrelated': nan}
    24200 {'loss': 0.7818241119384766, 'nll_fused_A': 0.26648640632629395, 'nll_fused_BinA': 0.2647836208343506, 'nll_marg_A': 1.456981897354126, 'nll_marg_BinA_only': 1.4642014503479004, 'fused_vs_marginal_gap': 1.1994178295135498, 'gap_related': 1.1994178295135498, 'gap_unrelated': nan}
    24300 {'loss': 0.9683496356010437, 'nll_fused_A': 0.386381059885025, 'nll_fused_BinA': 0.3836841583251953, 'nll_marg_A': 1.5625038146972656, 'nll_marg_BinA_only': 1.5621817111968994, 'fused_vs_marginal_gap': 1.178497552871704, 'gap_related': 1.178497552871704, 'gap_unrelated': nan}
    24400 {'loss': 0.8005118370056152, 'nll_fused_A': 0.2807309329509735, 'nll_fused_BinA': 0.26853126287460327, 'nll_marg_A': 1.492537498474121, 'nll_marg_BinA_only': 1.4933208227157593, 'fused_vs_marginal_gap': 1.2247895002365112, 'gap_related': 1.2247896194458008, 'gap_unrelated': nan}
    24500 {'loss': 0.8070358037948608, 'nll_fused_A': 0.26999151706695557, 'nll_fused_BinA': 0.27228689193725586, 'nll_marg_A': 1.5125046968460083, 'nll_marg_BinA_only': 1.5035556554794312, 'fused_vs_marginal_gap': 1.2312686443328857, 'gap_related': 1.2312687635421753, 'gap_unrelated': nan}
    24600 {'loss': 0.8402441740036011, 'nll_fused_A': 0.29293960332870483, 'nll_fused_BinA': 0.3075108528137207, 'nll_marg_A': 1.4828380346298218, 'nll_marg_BinA_only': 1.4980781078338623, 'fused_vs_marginal_gap': 1.1905672550201416, 'gap_related': 1.1905673742294312, 'gap_unrelated': nan}
    24700 {'loss': 0.7674198150634766, 'nll_fused_A': 0.2532727122306824, 'nll_fused_BinA': 0.2464998960494995, 'nll_marg_A': 1.4831268787384033, 'nll_marg_BinA_only': 1.487997055053711, 'fused_vs_marginal_gap': 1.241497278213501, 'gap_related': 1.241497278213501, 'gap_unrelated': nan}
    24800 {'loss': 0.8235403299331665, 'nll_fused_A': 0.3045130968093872, 'nll_fused_BinA': 0.2867898941040039, 'nll_marg_A': 1.4846549034118652, 'nll_marg_BinA_only': 1.4810478687286377, 'fused_vs_marginal_gap': 1.1942579746246338, 'gap_related': 1.1942579746246338, 'gap_unrelated': nan}
    24900 {'loss': 0.9153801202774048, 'nll_fused_A': 0.35570046305656433, 'nll_fused_BinA': 0.35157641768455505, 'nll_marg_A': 1.5236449241638184, 'nll_marg_BinA_only': 1.5291574001312256, 'fused_vs_marginal_gap': 1.1775809526443481, 'gap_related': 1.1775810718536377, 'gap_unrelated': nan}
    25000 {'loss': 0.7138000130653381, 'nll_fused_A': 0.21306301653385162, 'nll_fused_BinA': 0.2109060287475586, 'nll_marg_A': 1.4632502794265747, 'nll_marg_BinA_only': 1.4657888412475586, 'fused_vs_marginal_gap': 1.2548826932907104, 'gap_related': 1.2548828125, 'gap_unrelated': nan}
    25100 {'loss': 0.7675673961639404, 'nll_fused_A': 0.2405705600976944, 'nll_fused_BinA': 0.24313655495643616, 'nll_marg_A': 1.507532000541687, 'nll_marg_BinA_only': 1.5029914379119873, 'fused_vs_marginal_gap': 1.259854793548584, 'gap_related': 1.259854793548584, 'gap_unrelated': nan}
    25200 {'loss': 0.8501442074775696, 'nll_fused_A': 0.2981465756893158, 'nll_fused_BinA': 0.2903153896331787, 'nll_marg_A': 1.5679492950439453, 'nll_marg_BinA_only': 1.5501348972320557, 'fused_vs_marginal_gap': 1.259819507598877, 'gap_related': 1.259819507598877, 'gap_unrelated': nan}
    25300 {'loss': 0.7796422839164734, 'nll_fused_A': 0.24492529034614563, 'nll_fused_BinA': 0.24938011169433594, 'nll_marg_A': 1.5226151943206787, 'nll_marg_BinA_only': 1.5247780084609985, 'fused_vs_marginal_gap': 1.2753980159759521, 'gap_related': 1.2753978967666626, 'gap_unrelated': nan}
    25400 {'loss': 0.8490363359451294, 'nll_fused_A': 0.2854539155960083, 'nll_fused_BinA': 0.2949705123901367, 'nll_marg_A': 1.561432123184204, 'nll_marg_BinA_only': 1.5569039583206177, 'fused_vs_marginal_gap': 1.2619333267211914, 'gap_related': 1.261933445930481, 'gap_unrelated': nan}
    25500 {'loss': 0.8227086067199707, 'nll_fused_A': 0.3035276532173157, 'nll_fused_BinA': 0.2831363379955292, 'nll_marg_A': 1.495046615600586, 'nll_marg_BinA_only': 1.486994981765747, 'fused_vs_marginal_gap': 1.2038586139678955, 'gap_related': 1.2038586139678955, 'gap_unrelated': nan}
    25600 {'loss': 0.8670105934143066, 'nll_fused_A': 0.307064950466156, 'nll_fused_BinA': 0.30707797408103943, 'nll_marg_A': 1.5593769550323486, 'nll_marg_BinA_only': 1.5544276237487793, 'fused_vs_marginal_gap': 1.2473496198654175, 'gap_related': 1.2473496198654175, 'gap_unrelated': nan}
    25700 {'loss': 0.8925060033798218, 'nll_fused_A': 0.3340115547180176, 'nll_fused_BinA': 0.3276175558567047, 'nll_marg_A': 1.5489497184753418, 'nll_marg_BinA_only': 1.5466545820236206, 'fused_vs_marginal_gap': 1.2190370559692383, 'gap_related': 1.2190370559692383, 'gap_unrelated': nan}
    25800 {'loss': 0.7195044159889221, 'nll_fused_A': 0.21525974571704865, 'nll_fused_BinA': 0.21728366613388062, 'nll_marg_A': 1.4588093757629395, 'nll_marg_BinA_only': 1.4404031038284302, 'fused_vs_marginal_gap': 1.2231194972991943, 'gap_related': 1.2231194972991943, 'gap_unrelated': nan}
    25900 {'loss': 0.7728859782218933, 'nll_fused_A': 0.2545933723449707, 'nll_fused_BinA': 0.25446760654449463, 'nll_marg_A': 1.4734677076339722, 'nll_marg_BinA_only': 1.4869434833526611, 'fused_vs_marginal_gap': 1.2324758768081665, 'gap_related': 1.232475757598877, 'gap_unrelated': nan}
    26000 {'loss': 0.782690703868866, 'nll_fused_A': 0.27835243940353394, 'nll_fused_BinA': 0.25006240606307983, 'nll_marg_A': 1.497075080871582, 'nll_marg_BinA_only': 1.4851101636886597, 'fused_vs_marginal_gap': 1.2350478172302246, 'gap_related': 1.235047698020935, 'gap_unrelated': nan}
    26100 {'loss': 0.8388286828994751, 'nll_fused_A': 0.30026477575302124, 'nll_fused_BinA': 0.29819053411483765, 'nll_marg_A': 1.5018622875213623, 'nll_marg_BinA_only': 1.5046405792236328, 'fused_vs_marginal_gap': 1.2064499855041504, 'gap_related': 1.20645010471344, 'gap_unrelated': nan}
    26200 {'loss': 0.7537307739257812, 'nll_fused_A': 0.23204347491264343, 'nll_fused_BinA': 0.23922184109687805, 'nll_marg_A': 1.4829862117767334, 'nll_marg_BinA_only': 1.494702696800232, 'fused_vs_marginal_gap': 1.2554807662963867, 'gap_related': 1.2554807662963867, 'gap_unrelated': nan}
    26300 {'loss': 0.9233701229095459, 'nll_fused_A': 0.35719549655914307, 'nll_fused_BinA': 0.35353365540504456, 'nll_marg_A': 1.5422593355178833, 'nll_marg_BinA_only': 1.545373797416687, 'fused_vs_marginal_gap': 1.1918401718139648, 'gap_related': 1.1918401718139648, 'gap_unrelated': nan}
    26400 {'loss': 0.6806085109710693, 'nll_fused_A': 0.20344232022762299, 'nll_fused_BinA': 0.18167510628700256, 'nll_marg_A': 1.4596689939498901, 'nll_marg_BinA_only': 1.4486099481582642, 'fused_vs_marginal_gap': 1.266934871673584, 'gap_related': 1.266934871673584, 'gap_unrelated': nan}
    26500 {'loss': 0.8323951363563538, 'nll_fused_A': 0.2976158857345581, 'nll_fused_BinA': 0.28642725944519043, 'nll_marg_A': 1.5222768783569336, 'nll_marg_BinA_only': 1.5198981761932373, 'fused_vs_marginal_gap': 1.2334709167480469, 'gap_related': 1.2334709167480469, 'gap_unrelated': nan}
    26600 {'loss': 0.6616383790969849, 'nll_fused_A': 0.17855127155780792, 'nll_fused_BinA': 0.16195496916770935, 'nll_marg_A': 1.4870600700378418, 'nll_marg_BinA_only': 1.4890620708465576, 'fused_vs_marginal_gap': 1.3271071910858154, 'gap_related': 1.3271071910858154, 'gap_unrelated': nan}
    26700 {'loss': 0.7623127698898315, 'nll_fused_A': 0.22896023094654083, 'nll_fused_BinA': 0.24328461289405823, 'nll_marg_A': 1.5011335611343384, 'nll_marg_BinA_only': 1.50496506690979, 'fused_vs_marginal_gap': 1.2616804838180542, 'gap_related': 1.2616803646087646, 'gap_unrelated': nan}
    26800 {'loss': 0.8116983771324158, 'nll_fused_A': 0.2788543403148651, 'nll_fused_BinA': 0.2760947346687317, 'nll_marg_A': 1.50649094581604, 'nll_marg_BinA_only': 1.4968301057815552, 'fused_vs_marginal_gap': 1.2207353115081787, 'gap_related': 1.2207353115081787, 'gap_unrelated': nan}
    26900 {'loss': 0.7625673413276672, 'nll_fused_A': 0.23501995205879211, 'nll_fused_BinA': 0.2449902445077896, 'nll_marg_A': 1.4902371168136597, 'nll_marg_BinA_only': 1.484067678451538, 'fused_vs_marginal_gap': 1.2390775680541992, 'gap_related': 1.2390774488449097, 'gap_unrelated': nan}
    27000 {'loss': 0.5908898115158081, 'nll_fused_A': 0.1268714964389801, 'nll_fused_BinA': 0.11923984438180923, 'nll_marg_A': 1.445294976234436, 'nll_marg_BinA_only': 1.4494152069091797, 'fused_vs_marginal_gap': 1.3301753997802734, 'gap_related': 1.330175518989563, 'gap_unrelated': nan}
    27100 {'loss': 0.7130080461502075, 'nll_fused_A': 0.19828584790229797, 'nll_fused_BinA': 0.20456281304359436, 'nll_marg_A': 1.4965314865112305, 'nll_marg_BinA_only': 1.5001566410064697, 'fused_vs_marginal_gap': 1.2955937385559082, 'gap_related': 1.2955937385559082, 'gap_unrelated': nan}
    27200 {'loss': 0.7615087032318115, 'nll_fused_A': 0.24441595375537872, 'nll_fused_BinA': 0.2441222071647644, 'nll_marg_A': 1.4802056550979614, 'nll_marg_BinA_only': 1.471655011177063, 'fused_vs_marginal_gap': 1.2275328636169434, 'gap_related': 1.2275328636169434, 'gap_unrelated': nan}
    27300 {'loss': 0.6737918853759766, 'nll_fused_A': 0.16740113496780396, 'nll_fused_BinA': 0.17836830019950867, 'nll_marg_A': 1.4840105772018433, 'nll_marg_BinA_only': 1.4884357452392578, 'fused_vs_marginal_gap': 1.3100676536560059, 'gap_related': 1.3100675344467163, 'gap_unrelated': nan}
    27400 {'loss': 0.7939965724945068, 'nll_fused_A': 0.2659624218940735, 'nll_fused_BinA': 0.26274800300598145, 'nll_marg_A': 1.504866123199463, 'nll_marg_BinA_only': 1.5101792812347412, 'fused_vs_marginal_gap': 1.2474312782287598, 'gap_related': 1.2474312782287598, 'gap_unrelated': nan}
    27500 {'loss': 0.7185975909233093, 'nll_fused_A': 0.19315706193447113, 'nll_fused_BinA': 0.20931999385356903, 'nll_marg_A': 1.5044348239898682, 'nll_marg_BinA_only': 1.5026471614837646, 'fused_vs_marginal_gap': 1.2933270931243896, 'gap_related': 1.2933270931243896, 'gap_unrelated': nan}
    27600 {'loss': 0.5785204172134399, 'nll_fused_A': 0.11232934892177582, 'nll_fused_BinA': 0.11042766273021698, 'nll_marg_A': 1.4479798078536987, 'nll_marg_BinA_only': 1.4497567415237427, 'fused_vs_marginal_gap': 1.3393291234970093, 'gap_related': 1.3393290042877197, 'gap_unrelated': nan}
    27700 {'loss': 0.6542015671730042, 'nll_fused_A': 0.15523794293403625, 'nll_fused_BinA': 0.15866602957248688, 'nll_marg_A': 1.4965472221374512, 'nll_marg_BinA_only': 1.5121941566467285, 'fused_vs_marginal_gap': 1.3535281419754028, 'gap_related': 1.3535281419754028, 'gap_unrelated': nan}
    27800 {'loss': 0.6557453870773315, 'nll_fused_A': 0.17432847619056702, 'nll_fused_BinA': 0.16738399863243103, 'nll_marg_A': 1.453542709350586, 'nll_marg_BinA_only': 1.4554047584533691, 'fused_vs_marginal_gap': 1.2880206108093262, 'gap_related': 1.2880206108093262, 'gap_unrelated': nan}
    27900 {'loss': 0.6795395016670227, 'nll_fused_A': 0.17748934030532837, 'nll_fused_BinA': 0.17590312659740448, 'nll_marg_A': 1.501298427581787, 'nll_marg_BinA_only': 1.496727705001831, 'fused_vs_marginal_gap': 1.3208246231079102, 'gap_related': 1.3208246231079102, 'gap_unrelated': nan}
    28000 {'loss': 0.6730084419250488, 'nll_fused_A': 0.17438118159770966, 'nll_fused_BinA': 0.17201557755470276, 'nll_marg_A': 1.4955949783325195, 'nll_marg_BinA_only': 1.4815168380737305, 'fused_vs_marginal_gap': 1.3095011711120605, 'gap_related': 1.30950129032135, 'gap_unrelated': nan}
    28100 {'loss': 0.5898439884185791, 'nll_fused_A': 0.12125321477651596, 'nll_fused_BinA': 0.11163356155157089, 'nll_marg_A': 1.4727815389633179, 'nll_marg_BinA_only': 1.4596848487854004, 'fused_vs_marginal_gap': 1.3480510711669922, 'gap_related': 1.3480513095855713, 'gap_unrelated': nan}
    28200 {'loss': 0.5809911489486694, 'nll_fused_A': 0.10035905987024307, 'nll_fused_BinA': 0.11079404503107071, 'nll_marg_A': 1.4669644832611084, 'nll_marg_BinA_only': 1.4579436779022217, 'fused_vs_marginal_gap': 1.3471496105194092, 'gap_related': 1.3471496105194092, 'gap_unrelated': nan}
    28300 {'loss': 0.7487666606903076, 'nll_fused_A': 0.2270459085702896, 'nll_fused_BinA': 0.22083953022956848, 'nll_marg_A': 1.5327110290527344, 'nll_marg_BinA_only': 1.5300343036651611, 'fused_vs_marginal_gap': 1.3091946840286255, 'gap_related': 1.3091946840286255, 'gap_unrelated': nan}
    28400 {'loss': 0.6665270328521729, 'nll_fused_A': 0.1562320441007614, 'nll_fused_BinA': 0.1744426041841507, 'nll_marg_A': 1.4840493202209473, 'nll_marg_BinA_only': 1.4873275756835938, 'fused_vs_marginal_gap': 1.3128849267959595, 'gap_related': 1.3128849267959595, 'gap_unrelated': nan}
    28500 {'loss': 0.6651667356491089, 'nll_fused_A': 0.1569145917892456, 'nll_fused_BinA': 0.17243072390556335, 'nll_marg_A': 1.4855387210845947, 'nll_marg_BinA_only': 1.4935107231140137, 'fused_vs_marginal_gap': 1.321079969406128, 'gap_related': 1.3210798501968384, 'gap_unrelated': nan}
    28600 {'loss': 0.7173779010772705, 'nll_fused_A': 0.19894251227378845, 'nll_fused_BinA': 0.1973443329334259, 'nll_marg_A': 1.5345027446746826, 'nll_marg_BinA_only': 1.5183300971984863, 'fused_vs_marginal_gap': 1.3209856748580933, 'gap_related': 1.3209855556488037, 'gap_unrelated': nan}
    28700 {'loss': 0.7054018974304199, 'nll_fused_A': 0.19831988215446472, 'nll_fused_BinA': 0.20616495609283447, 'nll_marg_A': 1.4658031463623047, 'nll_marg_BinA_only': 1.4593207836151123, 'fused_vs_marginal_gap': 1.2531559467315674, 'gap_related': 1.2531559467315674, 'gap_unrelated': nan}
    28800 {'loss': 0.6694778203964233, 'nll_fused_A': 0.17622795701026917, 'nll_fused_BinA': 0.17538893222808838, 'nll_marg_A': 1.4707348346710205, 'nll_marg_BinA_only': 1.4883027076721191, 'fused_vs_marginal_gap': 1.3129137754440308, 'gap_related': 1.3129137754440308, 'gap_unrelated': nan}
    28900 {'loss': 0.6655322909355164, 'nll_fused_A': 0.1661524623632431, 'nll_fused_BinA': 0.1649344563484192, 'nll_marg_A': 1.502506971359253, 'nll_marg_BinA_only': 1.4983680248260498, 'fused_vs_marginal_gap': 1.3334336280822754, 'gap_related': 1.3334336280822754, 'gap_unrelated': nan}
    29000 {'loss': 0.7570820450782776, 'nll_fused_A': 0.22219198942184448, 'nll_fused_BinA': 0.2431582659482956, 'nll_marg_A': 1.49088716506958, 'nll_marg_BinA_only': 1.505030632019043, 'fused_vs_marginal_gap': 1.2618722915649414, 'gap_related': 1.2618722915649414, 'gap_unrelated': nan}
    29100 {'loss': 0.6254668235778809, 'nll_fused_A': 0.12907323241233826, 'nll_fused_BinA': 0.13328300416469574, 'nll_marg_A': 1.5115394592285156, 'nll_marg_BinA_only': 1.5123069286346436, 'fused_vs_marginal_gap': 1.3790239095687866, 'gap_related': 1.3790240287780762, 'gap_unrelated': nan}
    29200 {'loss': 0.5838465094566345, 'nll_fused_A': 0.10384492576122284, 'nll_fused_BinA': 0.10459772497415543, 'nll_marg_A': 1.4936509132385254, 'nll_marg_BinA_only': 1.4917328357696533, 'fused_vs_marginal_gap': 1.3871351480484009, 'gap_related': 1.3871352672576904, 'gap_unrelated': nan}
    29300 {'loss': 0.6300176382064819, 'nll_fused_A': 0.1387341320514679, 'nll_fused_BinA': 0.14340871572494507, 'nll_marg_A': 1.4832955598831177, 'nll_marg_BinA_only': 1.4663782119750977, 'fused_vs_marginal_gap': 1.3229694366455078, 'gap_related': 1.3229694366455078, 'gap_unrelated': nan}
    29400 {'loss': 0.6224759817123413, 'nll_fused_A': 0.1445155292749405, 'nll_fused_BinA': 0.13107915222644806, 'nll_marg_A': 1.493473768234253, 'nll_marg_BinA_only': 1.4862264394760132, 'fused_vs_marginal_gap': 1.3551472425460815, 'gap_related': 1.3551472425460815, 'gap_unrelated': nan}
    29500 {'loss': 0.5735417008399963, 'nll_fused_A': 0.09636880457401276, 'nll_fused_BinA': 0.0939832478761673, 'nll_marg_A': 1.5021592378616333, 'nll_marg_BinA_only': 1.4960473775863647, 'fused_vs_marginal_gap': 1.4020642042160034, 'gap_related': 1.4020642042160034, 'gap_unrelated': nan}
    29600 {'loss': 0.649422287940979, 'nll_fused_A': 0.14497032761573792, 'nll_fused_BinA': 0.14780940115451813, 'nll_marg_A': 1.527072548866272, 'nll_marg_BinA_only': 1.5450804233551025, 'fused_vs_marginal_gap': 1.397270917892456, 'gap_related': 1.397270917892456, 'gap_unrelated': nan}
    29700 {'loss': 0.6784735321998596, 'nll_fused_A': 0.18290388584136963, 'nll_fused_BinA': 0.17875942587852478, 'nll_marg_A': 1.4828097820281982, 'nll_marg_BinA_only': 1.4994487762451172, 'fused_vs_marginal_gap': 1.3206894397735596, 'gap_related': 1.3206894397735596, 'gap_unrelated': nan}
    29800 {'loss': 0.6628671884536743, 'nll_fused_A': 0.17246952652931213, 'nll_fused_BinA': 0.1690748631954193, 'nll_marg_A': 1.4735047817230225, 'nll_marg_BinA_only': 1.4838356971740723, 'fused_vs_marginal_gap': 1.3147609233856201, 'gap_related': 1.3147609233856201, 'gap_unrelated': nan}
    29900 {'loss': 0.5335620641708374, 'nll_fused_A': 0.06796541064977646, 'nll_fused_BinA': 0.07575127482414246, 'nll_marg_A': 1.4580703973770142, 'nll_marg_BinA_only': 1.455824375152588, 'fused_vs_marginal_gap': 1.3800731897354126, 'gap_related': 1.3800731897354126, 'gap_unrelated': nan}
    30000 {'loss': 0.4887773394584656, 'nll_fused_A': 0.03765082359313965, 'nll_fused_BinA': 0.03820614889264107, 'nll_marg_A': 1.4642530679702759, 'nll_marg_BinA_only': 1.454099178314209, 'fused_vs_marginal_gap': 1.4158930778503418, 'gap_related': 1.4158930778503418, 'gap_unrelated': nan}
    30100 {'loss': 0.5459976196289062, 'nll_fused_A': 0.08806723356246948, 'nll_fused_BinA': 0.07764680683612823, 'nll_marg_A': 1.47310209274292, 'nll_marg_BinA_only': 1.4690711498260498, 'fused_vs_marginal_gap': 1.3914244174957275, 'gap_related': 1.3914244174957275, 'gap_unrelated': nan}
    30200 {'loss': 0.5572634935379028, 'nll_fused_A': 0.08015021681785583, 'nll_fused_BinA': 0.0776187926530838, 'nll_marg_A': 1.5186653137207031, 'nll_marg_BinA_only': 1.5230625867843628, 'fused_vs_marginal_gap': 1.445443868637085, 'gap_related': 1.445443868637085, 'gap_unrelated': nan}
    30300 {'loss': 0.5246211290359497, 'nll_fused_A': 0.052288420498371124, 'nll_fused_BinA': 0.06282918155193329, 'nll_marg_A': 1.48701810836792, 'nll_marg_BinA_only': 1.4953243732452393, 'fused_vs_marginal_gap': 1.4324951171875, 'gap_related': 1.4324951171875, 'gap_unrelated': nan}
    30400 {'loss': 0.6407221555709839, 'nll_fused_A': 0.14800989627838135, 'nll_fused_BinA': 0.14093762636184692, 'nll_marg_A': 1.517938494682312, 'nll_marg_BinA_only': 1.5011286735534668, 'fused_vs_marginal_gap': 1.3601911067962646, 'gap_related': 1.3601911067962646, 'gap_unrelated': nan}
    30500 {'loss': 0.5574138760566711, 'nll_fused_A': 0.08265646547079086, 'nll_fused_BinA': 0.08855380862951279, 'nll_marg_A': 1.480210304260254, 'nll_marg_BinA_only': 1.4885399341583252, 'fused_vs_marginal_gap': 1.3999860286712646, 'gap_related': 1.3999861478805542, 'gap_unrelated': nan}
    30600 {'loss': 0.5500422716140747, 'nll_fused_A': 0.08150260150432587, 'nll_fused_BinA': 0.08777646720409393, 'nll_marg_A': 1.459383487701416, 'nll_marg_BinA_only': 1.4659597873687744, 'fused_vs_marginal_gap': 1.378183364868164, 'gap_related': 1.378183364868164, 'gap_unrelated': nan}
    30700 {'loss': 0.562142550945282, 'nll_fused_A': 0.0794239491224289, 'nll_fused_BinA': 0.08594167232513428, 'nll_marg_A': 1.507912278175354, 'nll_marg_BinA_only': 1.513824224472046, 'fused_vs_marginal_gap': 1.4278826713562012, 'gap_related': 1.4278825521469116, 'gap_unrelated': nan}
    30800 {'loss': 0.47153419256210327, 'nll_fused_A': 0.020627636462450027, 'nll_fused_BinA': 0.018703706562519073, 'nll_marg_A': 1.4888073205947876, 'nll_marg_BinA_only': 1.4949798583984375, 'fused_vs_marginal_gap': 1.476276159286499, 'gap_related': 1.476276159286499, 'gap_unrelated': nan}
    30900 {'loss': 0.45066630840301514, 'nll_fused_A': 0.0032604485750198364, 'nll_fused_BinA': 0.01872582733631134, 'nll_marg_A': 1.436540961265564, 'nll_marg_BinA_only': 1.4438793659210205, 'fused_vs_marginal_gap': 1.4251536130905151, 'gap_related': 1.4251534938812256, 'gap_unrelated': nan}
    31000 {'loss': 0.5780120491981506, 'nll_fused_A': 0.09584309160709381, 'nll_fused_BinA': 0.09546947479248047, 'nll_marg_A': 1.512632131576538, 'nll_marg_BinA_only': 1.5190293788909912, 'fused_vs_marginal_gap': 1.4235599040985107, 'gap_related': 1.4235599040985107, 'gap_unrelated': nan}
    31100 {'loss': 0.5122251510620117, 'nll_fused_A': 0.04682294279336929, 'nll_fused_BinA': 0.04948059469461441, 'nll_marg_A': 1.4956588745117188, 'nll_marg_BinA_only': 1.502126932144165, 'fused_vs_marginal_gap': 1.4526464939117432, 'gap_related': 1.4526463747024536, 'gap_unrelated': nan}
    31200 {'loss': 0.5604297518730164, 'nll_fused_A': 0.0910293310880661, 'nll_fused_BinA': 0.08552172034978867, 'nll_marg_A': 1.4919973611831665, 'nll_marg_BinA_only': 1.4899622201919556, 'fused_vs_marginal_gap': 1.4044404029846191, 'gap_related': 1.4044404029846191, 'gap_unrelated': nan}
    31300 {'loss': 0.430347204208374, 'nll_fused_A': -0.008658603765070438, 'nll_fused_BinA': -0.0007198471575975418, 'nll_marg_A': 1.4455487728118896, 'nll_marg_BinA_only': 1.4416579008102417, 'fused_vs_marginal_gap': 1.4423778057098389, 'gap_related': 1.4423778057098389, 'gap_unrelated': nan}
    31400 {'loss': 0.6642696857452393, 'nll_fused_A': 0.15328310430049896, 'nll_fused_BinA': 0.16017276048660278, 'nll_marg_A': 1.527039885520935, 'nll_marg_BinA_only': 1.5190370082855225, 'fused_vs_marginal_gap': 1.3588643074035645, 'gap_related': 1.3588643074035645, 'gap_unrelated': nan}
    31500 {'loss': 0.461308091878891, 'nll_fused_A': 0.010861419141292572, 'nll_fused_BinA': 0.012686870992183685, 'nll_marg_A': 1.4845426082611084, 'nll_marg_BinA_only': 1.4967416524887085, 'fused_vs_marginal_gap': 1.4840548038482666, 'gap_related': 1.484054684638977, 'gap_unrelated': nan}
    31600 {'loss': 0.5613209009170532, 'nll_fused_A': 0.09443970024585724, 'nll_fused_BinA': 0.1013026237487793, 'nll_marg_A': 1.4389543533325195, 'nll_marg_BinA_only': 1.4462604522705078, 'fused_vs_marginal_gap': 1.3449578285217285, 'gap_related': 1.3449578285217285, 'gap_unrelated': nan}
    31700 {'loss': 0.5415030717849731, 'nll_fused_A': 0.05669664964079857, 'nll_fused_BinA': 0.08020582050085068, 'nll_marg_A': 1.480960726737976, 'nll_marg_BinA_only': 1.500903844833374, 'fused_vs_marginal_gap': 1.4206979274749756, 'gap_related': 1.4206980466842651, 'gap_unrelated': nan}
    31800 {'loss': 0.5302848219871521, 'nll_fused_A': 0.06703229248523712, 'nll_fused_BinA': 0.07118813693523407, 'nll_marg_A': 1.4632899761199951, 'nll_marg_BinA_only': 1.4598770141601562, 'fused_vs_marginal_gap': 1.3886886835098267, 'gap_related': 1.3886888027191162, 'gap_unrelated': nan}
    31900 {'loss': 0.40623632073402405, 'nll_fused_A': -0.024539992213249207, 'nll_fused_BinA': -0.01870339922606945, 'nll_marg_A': 1.4410055875778198, 'nll_marg_BinA_only': 1.4250516891479492, 'fused_vs_marginal_gap': 1.4437551498413086, 'gap_related': 1.4437551498413086, 'gap_unrelated': nan}
    32000 {'loss': 0.4311376214027405, 'nll_fused_A': 0.010100518353283405, 'nll_fused_BinA': -0.002271907404065132, 'nll_marg_A': 1.4345979690551758, 'nll_marg_BinA_only': 1.429673433303833, 'fused_vs_marginal_gap': 1.4319453239440918, 'gap_related': 1.4319453239440918, 'gap_unrelated': nan}
    32100 {'loss': 0.4760374128818512, 'nll_fused_A': 0.0288129523396492, 'nll_fused_BinA': 0.030327683314681053, 'nll_marg_A': 1.4568860530853271, 'nll_marg_BinA_only': 1.4628944396972656, 'fused_vs_marginal_gap': 1.4325666427612305, 'gap_related': 1.4325666427612305, 'gap_unrelated': nan}
    32200 {'loss': 0.44133666157722473, 'nll_fused_A': -0.011533054523169994, 'nll_fused_BinA': -0.011271001771092415, 'nll_marg_A': 1.520225167274475, 'nll_marg_BinA_only': 1.5163618326187134, 'fused_vs_marginal_gap': 1.5276328325271606, 'gap_related': 1.5276328325271606, 'gap_unrelated': nan}
    32300 {'loss': 0.5247799158096313, 'nll_fused_A': 0.0647062212228775, 'nll_fused_BinA': 0.07072684168815613, 'nll_marg_A': 1.4488039016723633, 'nll_marg_BinA_only': 1.4671820402145386, 'fused_vs_marginal_gap': 1.39645516872406, 'gap_related': 1.3964550495147705, 'gap_unrelated': nan}
    32400 {'loss': 0.5431293845176697, 'nll_fused_A': 0.07467146217823029, 'nll_fused_BinA': 0.08397428691387177, 'nll_marg_A': 1.4558453559875488, 'nll_marg_BinA_only': 1.4592361450195312, 'fused_vs_marginal_gap': 1.3752617835998535, 'gap_related': 1.375261902809143, 'gap_unrelated': nan}
    32500 {'loss': 0.593207061290741, 'nll_fused_A': 0.10549938678741455, 'nll_fused_BinA': 0.1053132712841034, 'nll_marg_A': 1.520813226699829, 'nll_marg_BinA_only': 1.5203970670700073, 'fused_vs_marginal_gap': 1.4150837659835815, 'gap_related': 1.4150837659835815, 'gap_unrelated': nan}
    32600 {'loss': 0.5338419079780579, 'nll_fused_A': 0.05308770388364792, 'nll_fused_BinA': 0.06663849949836731, 'nll_marg_A': 1.5042569637298584, 'nll_marg_BinA_only': 1.506134033203125, 'fused_vs_marginal_gap': 1.4394954442977905, 'gap_related': 1.43949556350708, 'gap_unrelated': nan}
    32700 {'loss': 0.5428802371025085, 'nll_fused_A': 0.07975294440984726, 'nll_fused_BinA': 0.06506505608558655, 'nll_marg_A': 1.5129642486572266, 'nll_marg_BinA_only': 1.5128428936004639, 'fused_vs_marginal_gap': 1.4477777481079102, 'gap_related': 1.4477777481079102, 'gap_unrelated': nan}
    32800 {'loss': 0.40427762269973755, 'nll_fused_A': -0.03594892844557762, 'nll_fused_BinA': -0.02354050800204277, 'nll_marg_A': 1.4620091915130615, 'nll_marg_BinA_only': 1.4602248668670654, 'fused_vs_marginal_gap': 1.4837653636932373, 'gap_related': 1.4837653636932373, 'gap_unrelated': nan}
    32900 {'loss': 0.520088791847229, 'nll_fused_A': 0.061089009046554565, 'nll_fused_BinA': 0.046378277242183685, 'nll_marg_A': 1.5179460048675537, 'nll_marg_BinA_only': 1.5113003253936768, 'fused_vs_marginal_gap': 1.4649219512939453, 'gap_related': 1.4649219512939453, 'gap_unrelated': nan}
    33000 {'loss': 0.48722776770591736, 'nll_fused_A': 0.0038182903081178665, 'nll_fused_BinA': 0.022844728082418442, 'nll_marg_A': 1.5441250801086426, 'nll_marg_BinA_only': 1.5355031490325928, 'fused_vs_marginal_gap': 1.5126583576202393, 'gap_related': 1.5126583576202393, 'gap_unrelated': nan}
    33100 {'loss': 0.4065324664115906, 'nll_fused_A': -0.02442457713186741, 'nll_fused_BinA': -0.016491172835230827, 'nll_marg_A': 1.4345033168792725, 'nll_marg_BinA_only': 1.422633171081543, 'fused_vs_marginal_gap': 1.439124345779419, 'gap_related': 1.439124345779419, 'gap_unrelated': nan}
    33200 {'loss': 0.529843807220459, 'nll_fused_A': 0.06151469796895981, 'nll_fused_BinA': 0.05679517239332199, 'nll_marg_A': 1.5153141021728516, 'nll_marg_BinA_only': 1.5171699523925781, 'fused_vs_marginal_gap': 1.4603747129440308, 'gap_related': 1.4603745937347412, 'gap_unrelated': nan}
    33300 {'loss': 0.4318583905696869, 'nll_fused_A': -0.0001431647688150406, 'nll_fused_BinA': -0.01859942451119423, 'nll_marg_A': 1.50166916847229, 'nll_marg_BinA_only': 1.4987505674362183, 'fused_vs_marginal_gap': 1.5173499584197998, 'gap_related': 1.5173499584197998, 'gap_unrelated': nan}
    33400 {'loss': 0.3738120496273041, 'nll_fused_A': -0.048895418643951416, 'nll_fused_BinA': -0.05379754304885864, 'nll_marg_A': 1.474260687828064, 'nll_marg_BinA_only': 1.4676438570022583, 'fused_vs_marginal_gap': 1.5214413404464722, 'gap_related': 1.5214414596557617, 'gap_unrelated': nan}
    33500 {'loss': 0.4278649389743805, 'nll_fused_A': -0.01003163494169712, 'nll_fused_BinA': -0.017734523862600327, 'nll_marg_A': 1.4953631162643433, 'nll_marg_BinA_only': 1.4892593622207642, 'fused_vs_marginal_gap': 1.5069940090179443, 'gap_related': 1.5069938898086548, 'gap_unrelated': nan}
    33600 {'loss': 0.47515490651130676, 'nll_fused_A': 0.03658344969153404, 'nll_fused_BinA': 0.0309864804148674, 'nll_marg_A': 1.443977952003479, 'nll_marg_BinA_only': 1.4365909099578857, 'fused_vs_marginal_gap': 1.405604362487793, 'gap_related': 1.405604362487793, 'gap_unrelated': nan}
    33700 {'loss': 0.4015495777130127, 'nll_fused_A': -0.024838782846927643, 'nll_fused_BinA': -0.025169428437948227, 'nll_marg_A': 1.4472354650497437, 'nll_marg_BinA_only': 1.4364495277404785, 'fused_vs_marginal_gap': 1.4616191387176514, 'gap_related': 1.4616191387176514, 'gap_unrelated': nan}
    33800 {'loss': 0.5785791873931885, 'nll_fused_A': 0.09370412677526474, 'nll_fused_BinA': 0.10144835710525513, 'nll_marg_A': 1.4967318773269653, 'nll_marg_BinA_only': 1.488685131072998, 'fused_vs_marginal_gap': 1.3872368335723877, 'gap_related': 1.3872368335723877, 'gap_unrelated': nan}
    33900 {'loss': 0.41739460825920105, 'nll_fused_A': -0.028798455372452736, 'nll_fused_BinA': -0.020267747342586517, 'nll_marg_A': 1.4876729249954224, 'nll_marg_BinA_only': 1.483126163482666, 'fused_vs_marginal_gap': 1.5033938884735107, 'gap_related': 1.5033938884735107, 'gap_unrelated': nan}
    34000 {'loss': 0.502738893032074, 'nll_fused_A': 0.03461754694581032, 'nll_fused_BinA': 0.046868883073329926, 'nll_marg_A': 1.4849491119384766, 'nll_marg_BinA_only': 1.495018720626831, 'fused_vs_marginal_gap': 1.4481499195098877, 'gap_related': 1.4481499195098877, 'gap_unrelated': nan}
    34100 {'loss': 0.43673208355903625, 'nll_fused_A': -0.009814320132136345, 'nll_fused_BinA': -0.010100942105054855, 'nll_marg_A': 1.4992575645446777, 'nll_marg_BinA_only': 1.5107858180999756, 'fused_vs_marginal_gap': 1.5208866596221924, 'gap_related': 1.520886778831482, 'gap_unrelated': nan}
    34200 {'loss': 0.4618813991546631, 'nll_fused_A': 0.004999879747629166, 'nll_fused_BinA': 0.005197461694478989, 'nll_marg_A': 1.517279863357544, 'nll_marg_BinA_only': 1.4964348077774048, 'fused_vs_marginal_gap': 1.4912374019622803, 'gap_related': 1.4912374019622803, 'gap_unrelated': nan}
    34300 {'loss': 0.46302860975265503, 'nll_fused_A': -0.002825673669576645, 'nll_fused_BinA': 0.015728985890746117, 'nll_marg_A': 1.4938243627548218, 'nll_marg_BinA_only': 1.4933786392211914, 'fused_vs_marginal_gap': 1.4776495695114136, 'gap_related': 1.4776496887207031, 'gap_unrelated': nan}
    34400 {'loss': 0.33883213996887207, 'nll_fused_A': -0.06913808733224869, 'nll_fused_BinA': -0.0649840384721756, 'nll_marg_A': 1.415191888809204, 'nll_marg_BinA_only': 1.417104959487915, 'fused_vs_marginal_gap': 1.4820890426635742, 'gap_related': 1.4820890426635742, 'gap_unrelated': nan}
    34500 {'loss': 0.3882458209991455, 'nll_fused_A': -0.03312224894762039, 'nll_fused_BinA': -0.04082600027322769, 'nll_marg_A': 1.4633616209030151, 'nll_marg_BinA_only': 1.4634907245635986, 'fused_vs_marginal_gap': 1.504316806793213, 'gap_related': 1.504316806793213, 'gap_unrelated': nan}
    34600 {'loss': 0.43606603145599365, 'nll_fused_A': -0.006555207073688507, 'nll_fused_BinA': -0.0013314969837665558, 'nll_marg_A': 1.4645469188690186, 'nll_marg_BinA_only': 1.462572693824768, 'fused_vs_marginal_gap': 1.4639040231704712, 'gap_related': 1.4639040231704712, 'gap_unrelated': nan}
    34700 {'loss': 0.35961824655532837, 'nll_fused_A': -0.04976525157690048, 'nll_fused_BinA': -0.058261968195438385, 'nll_marg_A': 1.4426991939544678, 'nll_marg_BinA_only': 1.4449653625488281, 'fused_vs_marginal_gap': 1.5032272338867188, 'gap_related': 1.5032272338867188, 'gap_unrelated': nan}
    34800 {'loss': 0.32087773084640503, 'nll_fused_A': -0.07516898214817047, 'nll_fused_BinA': -0.08012422919273376, 'nll_marg_A': 1.4118421077728271, 'nll_marg_BinA_only': 1.415547490119934, 'fused_vs_marginal_gap': 1.4956717491149902, 'gap_related': 1.4956717491149902, 'gap_unrelated': nan}
    34900 {'loss': 0.40977349877357483, 'nll_fused_A': -0.024839846417307854, 'nll_fused_BinA': -0.028360776603221893, 'nll_marg_A': 1.4852874279022217, 'nll_marg_BinA_only': 1.4746330976486206, 'fused_vs_marginal_gap': 1.5029939413070679, 'gap_related': 1.5029940605163574, 'gap_unrelated': nan}
    35000 {'loss': 0.39912524819374084, 'nll_fused_A': -0.024069204926490784, 'nll_fused_BinA': -0.03391522914171219, 'nll_marg_A': 1.4675374031066895, 'nll_marg_BinA_only': 1.4725713729858398, 'fused_vs_marginal_gap': 1.5064866542816162, 'gap_related': 1.5064865350723267, 'gap_unrelated': nan}
    35100 {'loss': 0.3544539213180542, 'nll_fused_A': -0.06614742428064346, 'nll_fused_BinA': -0.06586067378520966, 'nll_marg_A': 1.467195987701416, 'nll_marg_BinA_only': 1.4502300024032593, 'fused_vs_marginal_gap': 1.5160906314849854, 'gap_related': 1.5160906314849854, 'gap_unrelated': nan}
    35200 {'loss': 0.2612919807434082, 'nll_fused_A': -0.1279171258211136, 'nll_fused_BinA': -0.1385158896446228, 'nll_marg_A': 1.460610032081604, 'nll_marg_BinA_only': 1.4498112201690674, 'fused_vs_marginal_gap': 1.588327169418335, 'gap_related': 1.588327169418335, 'gap_unrelated': nan}
    35300 {'loss': 0.30400440096855164, 'nll_fused_A': -0.10156743973493576, 'nll_fused_BinA': -0.10518243908882141, 'nll_marg_A': 1.4655234813690186, 'nll_marg_BinA_only': 1.4585745334625244, 'fused_vs_marginal_gap': 1.5637569427490234, 'gap_related': 1.5637569427490234, 'gap_unrelated': nan}
    35400 {'loss': 0.3669185936450958, 'nll_fused_A': -0.043269239366054535, 'nll_fused_BinA': -0.05602084845304489, 'nll_marg_A': 1.453067421913147, 'nll_marg_BinA_only': 1.4449189901351929, 'fused_vs_marginal_gap': 1.5009398460388184, 'gap_related': 1.5009398460388184, 'gap_unrelated': nan}
    35500 {'loss': 0.36071527004241943, 'nll_fused_A': -0.06187095493078232, 'nll_fused_BinA': -0.05066476762294769, 'nll_marg_A': 1.4331376552581787, 'nll_marg_BinA_only': 1.446903944015503, 'fused_vs_marginal_gap': 1.4975686073303223, 'gap_related': 1.4975686073303223, 'gap_unrelated': nan}
    35600 {'loss': 0.34477663040161133, 'nll_fused_A': -0.06371483206748962, 'nll_fused_BinA': -0.07123477011919022, 'nll_marg_A': 1.4504194259643555, 'nll_marg_BinA_only': 1.452700138092041, 'fused_vs_marginal_gap': 1.5239348411560059, 'gap_related': 1.5239348411560059, 'gap_unrelated': nan}
    35700 {'loss': 0.33866143226623535, 'nll_fused_A': -0.057601120322942734, 'nll_fused_BinA': -0.06458555161952972, 'nll_marg_A': 1.4017577171325684, 'nll_marg_BinA_only': 1.403398036956787, 'fused_vs_marginal_gap': 1.4679834842681885, 'gap_related': 1.4679834842681885, 'gap_unrelated': nan}
    35800 {'loss': 0.37958136200904846, 'nll_fused_A': -0.04699700325727463, 'nll_fused_BinA': -0.04082447290420532, 'nll_marg_A': 1.4483497142791748, 'nll_marg_BinA_only': 1.450348973274231, 'fused_vs_marginal_gap': 1.491173505783081, 'gap_related': 1.491173505783081, 'gap_unrelated': nan}
    35900 {'loss': 0.36206603050231934, 'nll_fused_A': -0.08403090387582779, 'nll_fused_BinA': -0.06313195824623108, 'nll_marg_A': 1.5013574361801147, 'nll_marg_BinA_only': 1.512603759765625, 'fused_vs_marginal_gap': 1.5757356882095337, 'gap_related': 1.5757358074188232, 'gap_unrelated': nan}
    36000 {'loss': 0.4042944312095642, 'nll_fused_A': -0.022499844431877136, 'nll_fused_BinA': -0.01571212336421013, 'nll_marg_A': 1.4225215911865234, 'nll_marg_BinA_only': 1.4296398162841797, 'fused_vs_marginal_gap': 1.4453518390655518, 'gap_related': 1.4453518390655518, 'gap_unrelated': nan}
    36100 {'loss': 0.36085012555122375, 'nll_fused_A': -0.05752161890268326, 'nll_fused_BinA': -0.06561391800642014, 'nll_marg_A': 1.479068398475647, 'nll_marg_BinA_only': 1.4829328060150146, 'fused_vs_marginal_gap': 1.5485467910766602, 'gap_related': 1.5485467910766602, 'gap_unrelated': nan}
    36200 {'loss': 0.3266199827194214, 'nll_fused_A': -0.06392259895801544, 'nll_fused_BinA': -0.08261655271053314, 'nll_marg_A': 1.428044319152832, 'nll_marg_BinA_only': 1.410444974899292, 'fused_vs_marginal_gap': 1.4930616617202759, 'gap_related': 1.4930616617202759, 'gap_unrelated': nan}
    36300 {'loss': 0.2622888386249542, 'nll_fused_A': -0.12578555941581726, 'nll_fused_BinA': -0.12087947875261307, 'nll_marg_A': 1.4030132293701172, 'nll_marg_BinA_only': 1.424180507659912, 'fused_vs_marginal_gap': 1.5450599193572998, 'gap_related': 1.5450599193572998, 'gap_unrelated': nan}
    36400 {'loss': 0.30832862854003906, 'nll_fused_A': -0.08644387125968933, 'nll_fused_BinA': -0.09877470135688782, 'nll_marg_A': 1.4434549808502197, 'nll_marg_BinA_only': 1.4337881803512573, 'fused_vs_marginal_gap': 1.5325628519058228, 'gap_related': 1.5325627326965332, 'gap_unrelated': nan}
    36500 {'loss': 0.3427745997905731, 'nll_fused_A': -0.07320288568735123, 'nll_fused_BinA': -0.07672961801290512, 'nll_marg_A': 1.4715502262115479, 'nll_marg_BinA_only': 1.463147759437561, 'fused_vs_marginal_gap': 1.5398774147033691, 'gap_related': 1.5398774147033691, 'gap_unrelated': nan}
    36600 {'loss': 0.3992116153240204, 'nll_fused_A': -0.035323645919561386, 'nll_fused_BinA': -0.025372497737407684, 'nll_marg_A': 1.45060396194458, 'nll_marg_BinA_only': 1.459708571434021, 'fused_vs_marginal_gap': 1.4850809574127197, 'gap_related': 1.4850809574127197, 'gap_unrelated': nan}
    36700 {'loss': 0.3374825716018677, 'nll_fused_A': -0.07145964354276657, 'nll_fused_BinA': -0.08040812611579895, 'nll_marg_A': 1.4644285440444946, 'nll_marg_BinA_only': 1.468052625656128, 'fused_vs_marginal_gap': 1.5484607219696045, 'gap_related': 1.5484607219696045, 'gap_unrelated': nan}
    36800 {'loss': 0.25432857871055603, 'nll_fused_A': -0.13543610274791718, 'nll_fused_BinA': -0.13470590114593506, 'nll_marg_A': 1.4322175979614258, 'nll_marg_BinA_only': 1.438920497894287, 'fused_vs_marginal_gap': 1.5736262798309326, 'gap_related': 1.5736262798309326, 'gap_unrelated': nan}
    36900 {'loss': 0.32151877880096436, 'nll_fused_A': -0.09303876757621765, 'nll_fused_BinA': -0.08282241970300674, 'nll_marg_A': 1.4408427476882935, 'nll_marg_BinA_only': 1.4318335056304932, 'fused_vs_marginal_gap': 1.5146559476852417, 'gap_related': 1.5146559476852417, 'gap_unrelated': nan}
    37000 {'loss': 0.2676585912704468, 'nll_fused_A': -0.11587491631507874, 'nll_fused_BinA': -0.12680193781852722, 'nll_marg_A': 1.4307432174682617, 'nll_marg_BinA_only': 1.4222444295883179, 'fused_vs_marginal_gap': 1.549046277999878, 'gap_related': 1.549046277999878, 'gap_unrelated': nan}
    37100 {'loss': 0.4821203351020813, 'nll_fused_A': 0.02134416624903679, 'nll_fused_BinA': 0.02025696635246277, 'nll_marg_A': 1.5182002782821655, 'nll_marg_BinA_only': 1.5103280544281006, 'fused_vs_marginal_gap': 1.4900710582733154, 'gap_related': 1.4900710582733154, 'gap_unrelated': nan}
    37200 {'loss': 0.2118341326713562, 'nll_fused_A': -0.15235677361488342, 'nll_fused_BinA': -0.1584606170654297, 'nll_marg_A': 1.3866724967956543, 'nll_marg_BinA_only': 1.376731038093567, 'fused_vs_marginal_gap': 1.5351917743682861, 'gap_related': 1.5351917743682861, 'gap_unrelated': nan}
    37300 {'loss': 0.3958348035812378, 'nll_fused_A': -0.018087798729538918, 'nll_fused_BinA': -0.03304089605808258, 'nll_marg_A': 1.4476733207702637, 'nll_marg_BinA_only': 1.4462816715240479, 'fused_vs_marginal_gap': 1.4793225526809692, 'gap_related': 1.4793225526809692, 'gap_unrelated': nan}
    37400 {'loss': 0.3830749988555908, 'nll_fused_A': -0.04543730989098549, 'nll_fused_BinA': -0.043141722679138184, 'nll_marg_A': 1.466159701347351, 'nll_marg_BinA_only': 1.4707376956939697, 'fused_vs_marginal_gap': 1.5138795375823975, 'gap_related': 1.513879418373108, 'gap_unrelated': nan}
    37500 {'loss': 0.42610663175582886, 'nll_fused_A': -0.014041729271411896, 'nll_fused_BinA': -0.006905151531100273, 'nll_marg_A': 1.4574142694473267, 'nll_marg_BinA_only': 1.4537761211395264, 'fused_vs_marginal_gap': 1.4606813192367554, 'gap_related': 1.460681438446045, 'gap_unrelated': nan}
    37600 {'loss': 0.38017746806144714, 'nll_fused_A': -0.0350164957344532, 'nll_fused_BinA': -0.04317498579621315, 'nll_marg_A': 1.4461913108825684, 'nll_marg_BinA_only': 1.4270439147949219, 'fused_vs_marginal_gap': 1.4702188968658447, 'gap_related': 1.4702188968658447, 'gap_unrelated': nan}
    37700 {'loss': 0.31881198287010193, 'nll_fused_A': -0.08899566531181335, 'nll_fused_BinA': -0.10695946961641312, 'nll_marg_A': 1.5082337856292725, 'nll_marg_BinA_only': 1.501507043838501, 'fused_vs_marginal_gap': 1.608466625213623, 'gap_related': 1.608466625213623, 'gap_unrelated': nan}
    37800 {'loss': 0.394646555185318, 'nll_fused_A': -0.03245025873184204, 'nll_fused_BinA': -0.033349402248859406, 'nll_marg_A': 1.4591033458709717, 'nll_marg_BinA_only': 1.459404468536377, 'fused_vs_marginal_gap': 1.4927538633346558, 'gap_related': 1.4927538633346558, 'gap_unrelated': nan}
    37900 {'loss': 0.30427730083465576, 'nll_fused_A': -0.09476903080940247, 'nll_fused_BinA': -0.10897229611873627, 'nll_marg_A': 1.4722676277160645, 'nll_marg_BinA_only': 1.4563114643096924, 'fused_vs_marginal_gap': 1.5652837753295898, 'gap_related': 1.5652837753295898, 'gap_unrelated': nan}
    38000 {'loss': 0.2655220031738281, 'nll_fused_A': -0.14285019040107727, 'nll_fused_BinA': -0.12986359000205994, 'nll_marg_A': 1.4608020782470703, 'nll_marg_BinA_only': 1.4620469808578491, 'fused_vs_marginal_gap': 1.5919106006622314, 'gap_related': 1.5919106006622314, 'gap_unrelated': nan}
    38100 {'loss': 0.23530785739421844, 'nll_fused_A': -0.1459246128797531, 'nll_fused_BinA': -0.15408767759799957, 'nll_marg_A': 1.4439096450805664, 'nll_marg_BinA_only': 1.4600975513458252, 'fused_vs_marginal_gap': 1.614185094833374, 'gap_related': 1.614185094833374, 'gap_unrelated': nan}
    38200 {'loss': 0.27839308977127075, 'nll_fused_A': -0.12953078746795654, 'nll_fused_BinA': -0.12265203893184662, 'nll_marg_A': 1.4663478136062622, 'nll_marg_BinA_only': 1.4427043199539185, 'fused_vs_marginal_gap': 1.5653564929962158, 'gap_related': 1.5653563737869263, 'gap_unrelated': nan}
    38300 {'loss': 0.31267914175987244, 'nll_fused_A': -0.08595424890518188, 'nll_fused_BinA': -0.089805006980896, 'nll_marg_A': 1.4275680780410767, 'nll_marg_BinA_only': 1.4192233085632324, 'fused_vs_marginal_gap': 1.509028434753418, 'gap_related': 1.509028434753418, 'gap_unrelated': nan}
    38400 {'loss': 0.3031434118747711, 'nll_fused_A': -0.09515316784381866, 'nll_fused_BinA': -0.10262883454561234, 'nll_marg_A': 1.4477273225784302, 'nll_marg_BinA_only': 1.4548676013946533, 'fused_vs_marginal_gap': 1.5574963092803955, 'gap_related': 1.5574963092803955, 'gap_unrelated': nan}
    38500 {'loss': 0.2638660669326782, 'nll_fused_A': -0.12442062050104141, 'nll_fused_BinA': -0.13232511281967163, 'nll_marg_A': 1.4450578689575195, 'nll_marg_BinA_only': 1.4598205089569092, 'fused_vs_marginal_gap': 1.592145562171936, 'gap_related': 1.5921456813812256, 'gap_unrelated': nan}
    38600 {'loss': 0.3911679983139038, 'nll_fused_A': -0.03695511445403099, 'nll_fused_BinA': -0.04860774427652359, 'nll_marg_A': 1.5028741359710693, 'nll_marg_BinA_only': 1.4930927753448486, 'fused_vs_marginal_gap': 1.5417006015777588, 'gap_related': 1.5417006015777588, 'gap_unrelated': nan}
    38700 {'loss': 0.35602155327796936, 'nll_fused_A': -0.07494311034679413, 'nll_fused_BinA': -0.06515514850616455, 'nll_marg_A': 1.478865385055542, 'nll_marg_BinA_only': 1.4860334396362305, 'fused_vs_marginal_gap': 1.5511884689331055, 'gap_related': 1.551188588142395, 'gap_unrelated': nan}
    38800 {'loss': 0.3404141068458557, 'nll_fused_A': -0.06805221736431122, 'nll_fused_BinA': -0.08890470862388611, 'nll_marg_A': 1.4991148710250854, 'nll_marg_BinA_only': 1.476585865020752, 'fused_vs_marginal_gap': 1.565490484237671, 'gap_related': 1.565490484237671, 'gap_unrelated': nan}
    38900 {'loss': 0.31251561641693115, 'nll_fused_A': -0.10130854696035385, 'nll_fused_BinA': -0.10460297763347626, 'nll_marg_A': 1.491703748703003, 'nll_marg_BinA_only': 1.4899623394012451, 'fused_vs_marginal_gap': 1.5945652723312378, 'gap_related': 1.5945652723312378, 'gap_unrelated': nan}
    39000 {'loss': 0.3516213595867157, 'nll_fused_A': -0.0717313215136528, 'nll_fused_BinA': -0.07162385433912277, 'nll_marg_A': 1.482548713684082, 'nll_marg_BinA_only': 1.4736018180847168, 'fused_vs_marginal_gap': 1.5452258586883545, 'gap_related': 1.545225739479065, 'gap_unrelated': nan}
    39100 {'loss': 0.23324918746948242, 'nll_fused_A': -0.15875644981861115, 'nll_fused_BinA': -0.15295195579528809, 'nll_marg_A': 1.4460935592651367, 'nll_marg_BinA_only': 1.4368540048599243, 'fused_vs_marginal_gap': 1.5898058414459229, 'gap_related': 1.5898058414459229, 'gap_unrelated': nan}
    39200 {'loss': 0.13711762428283691, 'nll_fused_A': -0.21771708130836487, 'nll_fused_BinA': -0.22043612599372864, 'nll_marg_A': 1.4095628261566162, 'nll_marg_BinA_only': 1.4031827449798584, 'fused_vs_marginal_gap': 1.6236190795898438, 'gap_related': 1.6236189603805542, 'gap_unrelated': nan}
    39300 {'loss': 0.26936280727386475, 'nll_fused_A': -0.14445990324020386, 'nll_fused_BinA': -0.11854083836078644, 'nll_marg_A': 1.4374721050262451, 'nll_marg_BinA_only': 1.4404025077819824, 'fused_vs_marginal_gap': 1.558943271636963, 'gap_related': 1.558943271636963, 'gap_unrelated': nan}
    39400 {'loss': 0.2191842943429947, 'nll_fused_A': -0.1679818034172058, 'nll_fused_BinA': -0.1562448889017105, 'nll_marg_A': 1.41941237449646, 'nll_marg_BinA_only': 1.4238523244857788, 'fused_vs_marginal_gap': 1.5800971984863281, 'gap_related': 1.5800971984863281, 'gap_unrelated': nan}
    39500 {'loss': 0.21543750166893005, 'nll_fused_A': -0.17275026440620422, 'nll_fused_BinA': -0.16792985796928406, 'nll_marg_A': 1.450641393661499, 'nll_marg_BinA_only': 1.4683678150177002, 'fused_vs_marginal_gap': 1.6362977027893066, 'gap_related': 1.6362977027893066, 'gap_unrelated': nan}
    39600 {'loss': 0.3146760165691376, 'nll_fused_A': -0.0843624621629715, 'nll_fused_BinA': -0.09646829217672348, 'nll_marg_A': 1.454843521118164, 'nll_marg_BinA_only': 1.4461719989776611, 'fused_vs_marginal_gap': 1.5426404476165771, 'gap_related': 1.5426403284072876, 'gap_unrelated': nan}
    39700 {'loss': 0.2837258279323578, 'nll_fused_A': -0.13189035654067993, 'nll_fused_BinA': -0.12600141763687134, 'nll_marg_A': 1.497647762298584, 'nll_marg_BinA_only': 1.4760236740112305, 'fused_vs_marginal_gap': 1.6020252704620361, 'gap_related': 1.6020251512527466, 'gap_unrelated': nan}
    39800 {'loss': 0.29006290435791016, 'nll_fused_A': -0.11291459202766418, 'nll_fused_BinA': -0.10637697577476501, 'nll_marg_A': 1.4343807697296143, 'nll_marg_BinA_only': 1.4406425952911377, 'fused_vs_marginal_gap': 1.5470194816589355, 'gap_related': 1.5470194816589355, 'gap_unrelated': nan}
    39900 {'loss': 0.3066844344139099, 'nll_fused_A': -0.07979249954223633, 'nll_fused_BinA': -0.1035328358411789, 'nll_marg_A': 1.4471832513809204, 'nll_marg_BinA_only': 1.4436538219451904, 'fused_vs_marginal_gap': 1.5471866130828857, 'gap_related': 1.5471866130828857, 'gap_unrelated': nan}
    40000 {'loss': 0.20317724347114563, 'nll_fused_A': -0.17812584340572357, 'nll_fused_BinA': -0.17199277877807617, 'nll_marg_A': 1.4286925792694092, 'nll_marg_BinA_only': 1.4314219951629639, 'fused_vs_marginal_gap': 1.6034146547317505, 'gap_related': 1.6034146547317505, 'gap_unrelated': nan}
    40100 {'loss': 0.3438079357147217, 'nll_fused_A': -0.07125718891620636, 'nll_fused_BinA': -0.08872886002063751, 'nll_marg_A': 1.5130465030670166, 'nll_marg_BinA_only': 1.5093296766281128, 'fused_vs_marginal_gap': 1.5980584621429443, 'gap_related': 1.5980584621429443, 'gap_unrelated': nan}
    40200 {'loss': 0.264541894197464, 'nll_fused_A': -0.12934507429599762, 'nll_fused_BinA': -0.13091576099395752, 'nll_marg_A': 1.4475371837615967, 'nll_marg_BinA_only': 1.4427516460418701, 'fused_vs_marginal_gap': 1.573667287826538, 'gap_related': 1.573667287826538, 'gap_unrelated': nan}
    40300 {'loss': 0.4076191782951355, 'nll_fused_A': -0.024625293910503387, 'nll_fused_BinA': -0.022348657250404358, 'nll_marg_A': 1.4578514099121094, 'nll_marg_BinA_only': 1.4733028411865234, 'fused_vs_marginal_gap': 1.4956514835357666, 'gap_related': 1.4956514835357666, 'gap_unrelated': nan}
    40400 {'loss': 0.19204181432724, 'nll_fused_A': -0.19024956226348877, 'nll_fused_BinA': -0.17860546708106995, 'nll_marg_A': 1.4257404804229736, 'nll_marg_BinA_only': 1.4164268970489502, 'fused_vs_marginal_gap': 1.5950324535369873, 'gap_related': 1.5950323343276978, 'gap_unrelated': nan}
    40500 {'loss': 0.2989204525947571, 'nll_fused_A': -0.11536073684692383, 'nll_fused_BinA': -0.1107296422123909, 'nll_marg_A': 1.480860948562622, 'nll_marg_BinA_only': 1.4694187641143799, 'fused_vs_marginal_gap': 1.5801483392715454, 'gap_related': 1.580148458480835, 'gap_unrelated': nan}
    40600 {'loss': 0.24219098687171936, 'nll_fused_A': -0.14802980422973633, 'nll_fused_BinA': -0.1494673788547516, 'nll_marg_A': 1.4535576105117798, 'nll_marg_BinA_only': 1.4462668895721436, 'fused_vs_marginal_gap': 1.5957342386245728, 'gap_related': 1.5957342386245728, 'gap_unrelated': nan}
    40700 {'loss': 0.24195678532123566, 'nll_fused_A': -0.14054295420646667, 'nll_fused_BinA': -0.14604945480823517, 'nll_marg_A': 1.4338970184326172, 'nll_marg_BinA_only': 1.4337153434753418, 'fused_vs_marginal_gap': 1.579764723777771, 'gap_related': 1.5797646045684814, 'gap_unrelated': nan}
    40800 {'loss': 0.2715079188346863, 'nll_fused_A': -0.13143885135650635, 'nll_fused_BinA': -0.13024352490901947, 'nll_marg_A': 1.47061026096344, 'nll_marg_BinA_only': 1.4643168449401855, 'fused_vs_marginal_gap': 1.5945603847503662, 'gap_related': 1.5945603847503662, 'gap_unrelated': nan}
    40900 {'loss': 0.21408531069755554, 'nll_fused_A': -0.179022878408432, 'nll_fused_BinA': -0.16901373863220215, 'nll_marg_A': 1.456019639968872, 'nll_marg_BinA_only': 1.4517147541046143, 'fused_vs_marginal_gap': 1.6207284927368164, 'gap_related': 1.6207283735275269, 'gap_unrelated': nan}
    41000 {'loss': 0.12994135916233063, 'nll_fused_A': -0.22449976205825806, 'nll_fused_BinA': -0.2338455766439438, 'nll_marg_A': 1.4371229410171509, 'nll_marg_BinA_only': 1.4286985397338867, 'fused_vs_marginal_gap': 1.6625441312789917, 'gap_related': 1.6625441312789917, 'gap_unrelated': nan}
    41100 {'loss': 0.2743171751499176, 'nll_fused_A': -0.11720271408557892, 'nll_fused_BinA': -0.12595930695533752, 'nll_marg_A': 1.4514576196670532, 'nll_marg_BinA_only': 1.4740707874298096, 'fused_vs_marginal_gap': 1.6000301837921143, 'gap_related': 1.6000301837921143, 'gap_unrelated': nan}
    41200 {'loss': 0.3032553195953369, 'nll_fused_A': -0.10360810160636902, 'nll_fused_BinA': -0.10554851591587067, 'nll_marg_A': 1.4662874937057495, 'nll_marg_BinA_only': 1.4484927654266357, 'fused_vs_marginal_gap': 1.5540412664413452, 'gap_related': 1.5540411472320557, 'gap_unrelated': nan}
    41300 {'loss': 0.187704399228096, 'nll_fused_A': -0.1833379566669464, 'nll_fused_BinA': -0.190298929810524, 'nll_marg_A': 1.443349003791809, 'nll_marg_BinA_only': 1.4393279552459717, 'fused_vs_marginal_gap': 1.629626989364624, 'gap_related': 1.629626989364624, 'gap_unrelated': nan}
    41400 {'loss': 0.1608724147081375, 'nll_fused_A': -0.1982734203338623, 'nll_fused_BinA': -0.20152615010738373, 'nll_marg_A': 1.40626859664917, 'nll_marg_BinA_only': 1.4078001976013184, 'fused_vs_marginal_gap': 1.6093263626098633, 'gap_related': 1.6093264818191528, 'gap_unrelated': nan}
    41500 {'loss': 0.1811230480670929, 'nll_fused_A': -0.18078947067260742, 'nll_fused_BinA': -0.19220957159996033, 'nll_marg_A': 1.4252314567565918, 'nll_marg_BinA_only': 1.4337029457092285, 'fused_vs_marginal_gap': 1.6259124279022217, 'gap_related': 1.6259124279022217, 'gap_unrelated': nan}
    41600 {'loss': 0.187101811170578, 'nll_fused_A': -0.20946091413497925, 'nll_fused_BinA': -0.18976137042045593, 'nll_marg_A': 1.4656715393066406, 'nll_marg_BinA_only': 1.4643664360046387, 'fused_vs_marginal_gap': 1.654127836227417, 'gap_related': 1.6541277170181274, 'gap_unrelated': nan}
    41700 {'loss': 0.25031495094299316, 'nll_fused_A': -0.13708361983299255, 'nll_fused_BinA': -0.14648768305778503, 'nll_marg_A': 1.459758996963501, 'nll_marg_BinA_only': 1.4559226036071777, 'fused_vs_marginal_gap': 1.6024103164672852, 'gap_related': 1.6024103164672852, 'gap_unrelated': nan}
    41800 {'loss': 0.37009328603744507, 'nll_fused_A': -0.05641583353281021, 'nll_fused_BinA': -0.04957124590873718, 'nll_marg_A': 1.4552974700927734, 'nll_marg_BinA_only': 1.4481613636016846, 'fused_vs_marginal_gap': 1.4977327585220337, 'gap_related': 1.4977326393127441, 'gap_unrelated': nan}
    41900 {'loss': 0.13459044694900513, 'nll_fused_A': -0.23066933453083038, 'nll_fused_BinA': -0.22971725463867188, 'nll_marg_A': 1.445028305053711, 'nll_marg_BinA_only': 1.4251747131347656, 'fused_vs_marginal_gap': 1.6548919677734375, 'gap_related': 1.6548919677734375, 'gap_unrelated': nan}
    42000 {'loss': 0.22893276810646057, 'nll_fused_A': -0.1580253541469574, 'nll_fused_BinA': -0.15530318021774292, 'nll_marg_A': 1.4388117790222168, 'nll_marg_BinA_only': 1.4477059841156006, 'fused_vs_marginal_gap': 1.6030092239379883, 'gap_related': 1.6030092239379883, 'gap_unrelated': nan}
    42100 {'loss': 0.15442803502082825, 'nll_fused_A': -0.19274812936782837, 'nll_fused_BinA': -0.21669813990592957, 'nll_marg_A': 1.4298354387283325, 'nll_marg_BinA_only': 1.4132201671600342, 'fused_vs_marginal_gap': 1.6299183368682861, 'gap_related': 1.6299183368682861, 'gap_unrelated': nan}
    42200 {'loss': 0.2486308217048645, 'nll_fused_A': -0.14747297763824463, 'nll_fused_BinA': -0.14136600494384766, 'nll_marg_A': 1.4474623203277588, 'nll_marg_BinA_only': 1.4475988149642944, 'fused_vs_marginal_gap': 1.588964819908142, 'gap_related': 1.588964819908142, 'gap_unrelated': nan}
    42300 {'loss': 0.23667508363723755, 'nll_fused_A': -0.15995275974273682, 'nll_fused_BinA': -0.15726369619369507, 'nll_marg_A': 1.4730819463729858, 'nll_marg_BinA_only': 1.4772064685821533, 'fused_vs_marginal_gap': 1.6344702243804932, 'gap_related': 1.6344701051712036, 'gap_unrelated': nan}
    42400 {'loss': 0.26290005445480347, 'nll_fused_A': -0.12643086910247803, 'nll_fused_BinA': -0.12959173321723938, 'nll_marg_A': 1.434736728668213, 'nll_marg_BinA_only': 1.4359441995620728, 'fused_vs_marginal_gap': 1.5655360221862793, 'gap_related': 1.5655360221862793, 'gap_unrelated': nan}
    42500 {'loss': 0.19086605310440063, 'nll_fused_A': -0.20397743582725525, 'nll_fused_BinA': -0.1846894919872284, 'nll_marg_A': 1.4558292627334595, 'nll_marg_BinA_only': 1.4566617012023926, 'fused_vs_marginal_gap': 1.6413512229919434, 'gap_related': 1.6413512229919434, 'gap_unrelated': nan}
    42600 {'loss': 0.23396319150924683, 'nll_fused_A': -0.1416873037815094, 'nll_fused_BinA': -0.14289423823356628, 'nll_marg_A': 1.397878646850586, 'nll_marg_BinA_only': 1.405365228652954, 'fused_vs_marginal_gap': 1.5482594966888428, 'gap_related': 1.5482594966888428, 'gap_unrelated': nan}
    42700 {'loss': 0.2609090805053711, 'nll_fused_A': -0.11947302520275116, 'nll_fused_BinA': -0.1349417269229889, 'nll_marg_A': 1.4389755725860596, 'nll_marg_BinA_only': 1.4221582412719727, 'fused_vs_marginal_gap': 1.5570999383926392, 'gap_related': 1.5570998191833496, 'gap_unrelated': nan}
    42800 {'loss': 0.33947235345840454, 'nll_fused_A': -0.083505779504776, 'nll_fused_BinA': -0.07987163960933685, 'nll_marg_A': 1.4813190698623657, 'nll_marg_BinA_only': 1.4714398384094238, 'fused_vs_marginal_gap': 1.5513116121292114, 'gap_related': 1.5513114929199219, 'gap_unrelated': nan}
    42900 {'loss': 0.23761498928070068, 'nll_fused_A': -0.13867056369781494, 'nll_fused_BinA': -0.15136930346488953, 'nll_marg_A': 1.4352848529815674, 'nll_marg_BinA_only': 1.43445885181427, 'fused_vs_marginal_gap': 1.5858280658721924, 'gap_related': 1.5858280658721924, 'gap_unrelated': nan}
    43000 {'loss': 0.22142070531845093, 'nll_fused_A': -0.16410166025161743, 'nll_fused_BinA': -0.15719228982925415, 'nll_marg_A': 1.4261449575424194, 'nll_marg_BinA_only': 1.4125313758850098, 'fused_vs_marginal_gap': 1.5697236061096191, 'gap_related': 1.5697236061096191, 'gap_unrelated': nan}
    43100 {'loss': 0.14994698762893677, 'nll_fused_A': -0.21666646003723145, 'nll_fused_BinA': -0.21436789631843567, 'nll_marg_A': 1.4310493469238281, 'nll_marg_BinA_only': 1.4391629695892334, 'fused_vs_marginal_gap': 1.6535309553146362, 'gap_related': 1.6535308361053467, 'gap_unrelated': nan}
    43200 {'loss': 0.15850841999053955, 'nll_fused_A': -0.19945144653320312, 'nll_fused_BinA': -0.21591481566429138, 'nll_marg_A': 1.4475288391113281, 'nll_marg_BinA_only': 1.4226329326629639, 'fused_vs_marginal_gap': 1.6385477781295776, 'gap_related': 1.6385477781295776, 'gap_unrelated': nan}
    43300 {'loss': 0.2840232253074646, 'nll_fused_A': -0.1085192859172821, 'nll_fused_BinA': -0.11413099616765976, 'nll_marg_A': 1.4357000589370728, 'nll_marg_BinA_only': 1.4386978149414062, 'fused_vs_marginal_gap': 1.5528287887573242, 'gap_related': 1.5528287887573242, 'gap_unrelated': nan}
    43400 {'loss': 0.21081319451332092, 'nll_fused_A': -0.17448587715625763, 'nll_fused_BinA': -0.16852569580078125, 'nll_marg_A': 1.4389488697052002, 'nll_marg_BinA_only': 1.4447156190872192, 'fused_vs_marginal_gap': 1.61324143409729, 'gap_related': 1.613241195678711, 'gap_unrelated': nan}
    43500 {'loss': 0.1643516719341278, 'nll_fused_A': -0.19471797347068787, 'nll_fused_BinA': -0.1957591474056244, 'nll_marg_A': 1.3950873613357544, 'nll_marg_BinA_only': 1.4061851501464844, 'fused_vs_marginal_gap': 1.6019442081451416, 'gap_related': 1.6019443273544312, 'gap_unrelated': nan}
    43600 {'loss': 0.1981143057346344, 'nll_fused_A': -0.18066734075546265, 'nll_fused_BinA': -0.18303683400154114, 'nll_marg_A': 1.4511711597442627, 'nll_marg_BinA_only': 1.4570573568344116, 'fused_vs_marginal_gap': 1.6400941610336304, 'gap_related': 1.6400941610336304, 'gap_unrelated': nan}
    43700 {'loss': 0.21417509019374847, 'nll_fused_A': -0.17231005430221558, 'nll_fused_BinA': -0.16389738023281097, 'nll_marg_A': 1.4325515031814575, 'nll_marg_BinA_only': 1.4222257137298584, 'fused_vs_marginal_gap': 1.5861231088638306, 'gap_related': 1.5861231088638306, 'gap_unrelated': nan}
    43800 {'loss': 0.15030241012573242, 'nll_fused_A': -0.22175569832324982, 'nll_fused_BinA': -0.2152746021747589, 'nll_marg_A': 1.4403457641601562, 'nll_marg_BinA_only': 1.4339184761047363, 'fused_vs_marginal_gap': 1.649193286895752, 'gap_related': 1.6491931676864624, 'gap_unrelated': nan}
    43900 {'loss': 0.08660328388214111, 'nll_fused_A': -0.2531222105026245, 'nll_fused_BinA': -0.2552723288536072, 'nll_marg_A': 1.3927075862884521, 'nll_marg_BinA_only': 1.3763818740844727, 'fused_vs_marginal_gap': 1.6316542625427246, 'gap_related': 1.6316542625427246, 'gap_unrelated': nan}
    44000 {'loss': 0.1993681937456131, 'nll_fused_A': -0.19145435094833374, 'nll_fused_BinA': -0.17736460268497467, 'nll_marg_A': 1.4472302198410034, 'nll_marg_BinA_only': 1.4461982250213623, 'fused_vs_marginal_gap': 1.6235628128051758, 'gap_related': 1.6235628128051758, 'gap_unrelated': nan}
    44100 {'loss': 0.17287757992744446, 'nll_fused_A': -0.19532999396324158, 'nll_fused_BinA': -0.20811817049980164, 'nll_marg_A': 1.465315818786621, 'nll_marg_BinA_only': 1.4394255876541138, 'fused_vs_marginal_gap': 1.6475437879562378, 'gap_related': 1.6475437879562378, 'gap_unrelated': nan}
    44200 {'loss': 0.1465650498867035, 'nll_fused_A': -0.21244186162948608, 'nll_fused_BinA': -0.22182801365852356, 'nll_marg_A': 1.4404187202453613, 'nll_marg_BinA_only': 1.4378485679626465, 'fused_vs_marginal_gap': 1.6596765518188477, 'gap_related': 1.6596765518188477, 'gap_unrelated': nan}
    44300 {'loss': 0.16736465692520142, 'nll_fused_A': -0.22232533991336823, 'nll_fused_BinA': -0.21277061104774475, 'nll_marg_A': 1.4894428253173828, 'nll_marg_BinA_only': 1.4893875122070312, 'fused_vs_marginal_gap': 1.7021582126617432, 'gap_related': 1.7021580934524536, 'gap_unrelated': nan}
    44400 {'loss': 0.11312362551689148, 'nll_fused_A': -0.2520884573459625, 'nll_fused_BinA': -0.232937753200531, 'nll_marg_A': 1.4056262969970703, 'nll_marg_BinA_only': 1.4079270362854004, 'fused_vs_marginal_gap': 1.6408648490905762, 'gap_related': 1.6408648490905762, 'gap_unrelated': nan}
    44500 {'loss': 0.23114565014839172, 'nll_fused_A': -0.16719338297843933, 'nll_fused_BinA': -0.1590244472026825, 'nll_marg_A': 1.4677603244781494, 'nll_marg_BinA_only': 1.485124111175537, 'fused_vs_marginal_gap': 1.6441484689712524, 'gap_related': 1.6441484689712524, 'gap_unrelated': nan}
    44600 {'loss': 0.21132361888885498, 'nll_fused_A': -0.17357969284057617, 'nll_fused_BinA': -0.17717024683952332, 'nll_marg_A': 1.4685591459274292, 'nll_marg_BinA_only': 1.4620921611785889, 'fused_vs_marginal_gap': 1.6392624378204346, 'gap_related': 1.639262318611145, 'gap_unrelated': nan}
    44700 {'loss': 0.2721085846424103, 'nll_fused_A': -0.1418411135673523, 'nll_fused_BinA': -0.13509082794189453, 'nll_marg_A': 1.4991724491119385, 'nll_marg_BinA_only': 1.51759934425354, 'fused_vs_marginal_gap': 1.6526901721954346, 'gap_related': 1.6526901721954346, 'gap_unrelated': nan}
    44800 {'loss': 0.2144814133644104, 'nll_fused_A': -0.17362894117832184, 'nll_fused_BinA': -0.16653922200202942, 'nll_marg_A': 1.4436976909637451, 'nll_marg_BinA_only': 1.4562232494354248, 'fused_vs_marginal_gap': 1.6227624416351318, 'gap_related': 1.6227624416351318, 'gap_unrelated': nan}
    44900 {'loss': 0.13209876418113708, 'nll_fused_A': -0.22846177220344543, 'nll_fused_BinA': -0.22389006614685059, 'nll_marg_A': 1.4150911569595337, 'nll_marg_BinA_only': 1.40953528881073, 'fused_vs_marginal_gap': 1.6334253549575806, 'gap_related': 1.6334253549575806, 'gap_unrelated': nan}
    45000 {'loss': 0.14507006108760834, 'nll_fused_A': -0.21378755569458008, 'nll_fused_BinA': -0.22865115106105804, 'nll_marg_A': 1.4595248699188232, 'nll_marg_BinA_only': 1.4544885158538818, 'fused_vs_marginal_gap': 1.683139681816101, 'gap_related': 1.6831395626068115, 'gap_unrelated': nan}
    45100 {'loss': 0.16770336031913757, 'nll_fused_A': -0.20168110728263855, 'nll_fused_BinA': -0.2026928961277008, 'nll_marg_A': 1.4363353252410889, 'nll_marg_BinA_only': 1.4468249082565308, 'fused_vs_marginal_gap': 1.6495177745819092, 'gap_related': 1.6495177745819092, 'gap_unrelated': nan}
    45200 {'loss': 0.24144761264324188, 'nll_fused_A': -0.15656885504722595, 'nll_fused_BinA': -0.14775805175304413, 'nll_marg_A': 1.4539210796356201, 'nll_marg_BinA_only': 1.450248122215271, 'fused_vs_marginal_gap': 1.598006248474121, 'gap_related': 1.5980061292648315, 'gap_unrelated': nan}
    45300 {'loss': 0.09278516471385956, 'nll_fused_A': -0.27367067337036133, 'nll_fused_BinA': -0.24731023609638214, 'nll_marg_A': 1.4073219299316406, 'nll_marg_BinA_only': 1.4248569011688232, 'fused_vs_marginal_gap': 1.6721670627593994, 'gap_related': 1.6721670627593994, 'gap_unrelated': nan}
    45400 {'loss': 0.24608054757118225, 'nll_fused_A': -0.1474248468875885, 'nll_fused_BinA': -0.14965161681175232, 'nll_marg_A': 1.4665319919586182, 'nll_marg_BinA_only': 1.4813295602798462, 'fused_vs_marginal_gap': 1.630981206893921, 'gap_related': 1.630981206893921, 'gap_unrelated': nan}
    45500 {'loss': 0.2974487245082855, 'nll_fused_A': -0.09916520863771439, 'nll_fused_BinA': -0.0961274802684784, 'nll_marg_A': 1.411085844039917, 'nll_marg_BinA_only': 1.3988490104675293, 'fused_vs_marginal_gap': 1.49497652053833, 'gap_related': 1.4949764013290405, 'gap_unrelated': nan}
    45600 {'loss': 0.24996574223041534, 'nll_fused_A': -0.13826756179332733, 'nll_fused_BinA': -0.14854072034358978, 'nll_marg_A': 1.4666223526000977, 'nll_marg_BinA_only': 1.466567873954773, 'fused_vs_marginal_gap': 1.6151084899902344, 'gap_related': 1.615108609199524, 'gap_unrelated': nan}
    45700 {'loss': 0.1429159939289093, 'nll_fused_A': -0.2260747253894806, 'nll_fused_BinA': -0.221055269241333, 'nll_marg_A': 1.439312219619751, 'nll_marg_BinA_only': 1.4460954666137695, 'fused_vs_marginal_gap': 1.6671507358551025, 'gap_related': 1.6671507358551025, 'gap_unrelated': nan}
    45800 {'loss': 0.10381504893302917, 'nll_fused_A': -0.2600470185279846, 'nll_fused_BinA': -0.26047828793525696, 'nll_marg_A': 1.474358081817627, 'nll_marg_BinA_only': 1.469588279724121, 'fused_vs_marginal_gap': 1.7300665378570557, 'gap_related': 1.7300665378570557, 'gap_unrelated': nan}
    45900 {'loss': 0.16206848621368408, 'nll_fused_A': -0.20641010999679565, 'nll_fused_BinA': -0.20908960700035095, 'nll_marg_A': 1.443603754043579, 'nll_marg_BinA_only': 1.4476834535598755, 'fused_vs_marginal_gap': 1.6567729711532593, 'gap_related': 1.6567730903625488, 'gap_unrelated': nan}
    46000 {'loss': 0.25307637453079224, 'nll_fused_A': -0.13336262106895447, 'nll_fused_BinA': -0.14523354172706604, 'nll_marg_A': 1.4610623121261597, 'nll_marg_BinA_only': 1.4470539093017578, 'fused_vs_marginal_gap': 1.592287540435791, 'gap_related': 1.592287540435791, 'gap_unrelated': nan}
    46100 {'loss': 0.08489009737968445, 'nll_fused_A': -0.2616564929485321, 'nll_fused_BinA': -0.262245774269104, 'nll_marg_A': 1.418776035308838, 'nll_marg_BinA_only': 1.4143757820129395, 'fused_vs_marginal_gap': 1.676621437072754, 'gap_related': 1.6766215562820435, 'gap_unrelated': nan}
    46200 {'loss': 0.07791095972061157, 'nll_fused_A': -0.27676403522491455, 'nll_fused_BinA': -0.2716294527053833, 'nll_marg_A': 1.4418987035751343, 'nll_marg_BinA_only': 1.4535810947418213, 'fused_vs_marginal_gap': 1.7252106666564941, 'gap_related': 1.7252106666564941, 'gap_unrelated': nan}
    46300 {'loss': 0.1584797203540802, 'nll_fused_A': -0.20694397389888763, 'nll_fused_BinA': -0.20482435822486877, 'nll_marg_A': 1.4179575443267822, 'nll_marg_BinA_only': 1.4162060022354126, 'fused_vs_marginal_gap': 1.621030330657959, 'gap_related': 1.621030330657959, 'gap_unrelated': nan}
    46400 {'loss': 0.10607302188873291, 'nll_fused_A': -0.24604135751724243, 'nll_fused_BinA': -0.2555573582649231, 'nll_marg_A': 1.4514758586883545, 'nll_marg_BinA_only': 1.4500839710235596, 'fused_vs_marginal_gap': 1.705641269683838, 'gap_related': 1.705641269683838, 'gap_unrelated': nan}
    46500 {'loss': 0.24709603190422058, 'nll_fused_A': -0.131146639585495, 'nll_fused_BinA': -0.1375933289527893, 'nll_marg_A': 1.4134445190429688, 'nll_marg_BinA_only': 1.408828854560852, 'fused_vs_marginal_gap': 1.5464222431182861, 'gap_related': 1.5464222431182861, 'gap_unrelated': nan}
    46600 {'loss': 0.13168033957481384, 'nll_fused_A': -0.23014509677886963, 'nll_fused_BinA': -0.23288452625274658, 'nll_marg_A': 1.4453612565994263, 'nll_marg_BinA_only': 1.464568018913269, 'fused_vs_marginal_gap': 1.6974525451660156, 'gap_related': 1.6974525451660156, 'gap_unrelated': nan}
    46700 {'loss': 0.08384287357330322, 'nll_fused_A': -0.26059600710868835, 'nll_fused_BinA': -0.2634764313697815, 'nll_marg_A': 1.4183269739151, 'nll_marg_BinA_only': 1.4236478805541992, 'fused_vs_marginal_gap': 1.6871243715286255, 'gap_related': 1.687124252319336, 'gap_unrelated': nan}
    46800 {'loss': 0.09670621156692505, 'nll_fused_A': -0.263062983751297, 'nll_fused_BinA': -0.26273098587989807, 'nll_marg_A': 1.4611868858337402, 'nll_marg_BinA_only': 1.4847631454467773, 'fused_vs_marginal_gap': 1.747494101524353, 'gap_related': 1.7474942207336426, 'gap_unrelated': nan}
    46900 {'loss': 0.08708712458610535, 'nll_fused_A': -0.28434357047080994, 'nll_fused_BinA': -0.25076496601104736, 'nll_marg_A': 1.4105172157287598, 'nll_marg_BinA_only': 1.3974957466125488, 'fused_vs_marginal_gap': 1.6482605934143066, 'gap_related': 1.6482608318328857, 'gap_unrelated': nan}
    47000 {'loss': 0.07277059555053711, 'nll_fused_A': -0.28528445959091187, 'nll_fused_BinA': -0.26615822315216064, 'nll_marg_A': 1.4150471687316895, 'nll_marg_BinA_only': 1.406125545501709, 'fused_vs_marginal_gap': 1.6722837686538696, 'gap_related': 1.6722837686538696, 'gap_unrelated': nan}
    47100 {'loss': 0.08154803514480591, 'nll_fused_A': -0.2641156017780304, 'nll_fused_BinA': -0.25758153200149536, 'nll_marg_A': 1.394547462463379, 'nll_marg_BinA_only': 1.3882174491882324, 'fused_vs_marginal_gap': 1.6457990407943726, 'gap_related': 1.6457990407943726, 'gap_unrelated': nan}
    47200 {'loss': 0.1359027922153473, 'nll_fused_A': -0.22162804007530212, 'nll_fused_BinA': -0.23066505789756775, 'nll_marg_A': 1.4435209035873413, 'nll_marg_BinA_only': 1.4404206275939941, 'fused_vs_marginal_gap': 1.6710858345031738, 'gap_related': 1.6710857152938843, 'gap_unrelated': nan}
    47300 {'loss': 0.1774609386920929, 'nll_fused_A': -0.19432014226913452, 'nll_fused_BinA': -0.1900055706501007, 'nll_marg_A': 1.4192085266113281, 'nll_marg_BinA_only': 1.4198848009109497, 'fused_vs_marginal_gap': 1.609890341758728, 'gap_related': 1.609890341758728, 'gap_unrelated': nan}
    47400 {'loss': 0.12287858128547668, 'nll_fused_A': -0.21980945765972137, 'nll_fused_BinA': -0.23938485980033875, 'nll_marg_A': 1.4273542165756226, 'nll_marg_BinA_only': 1.4110522270202637, 'fused_vs_marginal_gap': 1.6504371166229248, 'gap_related': 1.6504371166229248, 'gap_unrelated': nan}
    47500 {'loss': -0.03015071153640747, 'nll_fused_A': -0.33392685651779175, 'nll_fused_BinA': -0.34148165583610535, 'nll_marg_A': 1.3716965913772583, 'nll_marg_BinA_only': 1.3573451042175293, 'fused_vs_marginal_gap': 1.698826789855957, 'gap_related': 1.698826789855957, 'gap_unrelated': nan}
    47600 {'loss': 0.06944584846496582, 'nll_fused_A': -0.27657634019851685, 'nll_fused_BinA': -0.2719297707080841, 'nll_marg_A': 1.4144949913024902, 'nll_marg_BinA_only': 1.4065539836883545, 'fused_vs_marginal_gap': 1.6784838438034058, 'gap_related': 1.6784838438034058, 'gap_unrelated': nan}
    47700 {'loss': 0.19781669974327087, 'nll_fused_A': -0.19040687382221222, 'nll_fused_BinA': -0.17780160903930664, 'nll_marg_A': 1.4424679279327393, 'nll_marg_BinA_only': 1.43168306350708, 'fused_vs_marginal_gap': 1.6094847917556763, 'gap_related': 1.6094846725463867, 'gap_unrelated': nan}
    47800 {'loss': 0.07269442081451416, 'nll_fused_A': -0.2708982229232788, 'nll_fused_BinA': -0.2817447781562805, 'nll_marg_A': 1.4523621797561646, 'nll_marg_BinA_only': 1.4486409425735474, 'fused_vs_marginal_gap': 1.7303857803344727, 'gap_related': 1.7303857803344727, 'gap_unrelated': nan}
    47900 {'loss': 0.13687677681446075, 'nll_fused_A': -0.22615918517112732, 'nll_fused_BinA': -0.24115188419818878, 'nll_marg_A': 1.4862546920776367, 'nll_marg_BinA_only': 1.4892630577087402, 'fused_vs_marginal_gap': 1.7304149866104126, 'gap_related': 1.730414867401123, 'gap_unrelated': nan}
    48000 {'loss': 0.12866073846817017, 'nll_fused_A': -0.22637544572353363, 'nll_fused_BinA': -0.2316114902496338, 'nll_marg_A': 1.4272828102111816, 'nll_marg_BinA_only': 1.422408938407898, 'fused_vs_marginal_gap': 1.6540205478668213, 'gap_related': 1.6540205478668213, 'gap_unrelated': nan}
    48100 {'loss': 0.03633663058280945, 'nll_fused_A': -0.28602173924446106, 'nll_fused_BinA': -0.3012058734893799, 'nll_marg_A': 1.411163330078125, 'nll_marg_BinA_only': 1.4039700031280518, 'fused_vs_marginal_gap': 1.7051758766174316, 'gap_related': 1.7051758766174316, 'gap_unrelated': nan}
    48200 {'loss': 0.10812857747077942, 'nll_fused_A': -0.23770254850387573, 'nll_fused_BinA': -0.24926698207855225, 'nll_marg_A': 1.4290211200714111, 'nll_marg_BinA_only': 1.4177730083465576, 'fused_vs_marginal_gap': 1.6670398712158203, 'gap_related': 1.6670399904251099, 'gap_unrelated': nan}
    48300 {'loss': 0.1141340583562851, 'nll_fused_A': -0.24072390794754028, 'nll_fused_BinA': -0.24333588778972626, 'nll_marg_A': 1.4322904348373413, 'nll_marg_BinA_only': 1.4367845058441162, 'fused_vs_marginal_gap': 1.6801203489303589, 'gap_related': 1.6801203489303589, 'gap_unrelated': nan}
    48400 {'loss': -0.038735032081604004, 'nll_fused_A': -0.35338324308395386, 'nll_fused_BinA': -0.35765913128852844, 'nll_marg_A': 1.4164636135101318, 'nll_marg_BinA_only': 1.4246684312820435, 'fused_vs_marginal_gap': 1.782327651977539, 'gap_related': 1.7823275327682495, 'gap_unrelated': nan}
    48500 {'loss': 0.13548630475997925, 'nll_fused_A': -0.22632381319999695, 'nll_fused_BinA': -0.22296786308288574, 'nll_marg_A': 1.4211710691452026, 'nll_marg_BinA_only': 1.4193201065063477, 'fused_vs_marginal_gap': 1.6422879695892334, 'gap_related': 1.6422879695892334, 'gap_unrelated': nan}
    48600 {'loss': 0.08775460720062256, 'nll_fused_A': -0.2540181279182434, 'nll_fused_BinA': -0.26456886529922485, 'nll_marg_A': 1.4284296035766602, 'nll_marg_BinA_only': 1.432561993598938, 'fused_vs_marginal_gap': 1.6971309185028076, 'gap_related': 1.6971309185028076, 'gap_unrelated': nan}
    48700 {'loss': 0.11186814308166504, 'nll_fused_A': -0.22366683185100555, 'nll_fused_BinA': -0.24311110377311707, 'nll_marg_A': 1.406930923461914, 'nll_marg_BinA_only': 1.4038150310516357, 'fused_vs_marginal_gap': 1.6469261646270752, 'gap_related': 1.6469261646270752, 'gap_unrelated': nan}
    48800 {'loss': 0.1790372133255005, 'nll_fused_A': -0.19818994402885437, 'nll_fused_BinA': -0.20104703307151794, 'nll_marg_A': 1.4651373624801636, 'nll_marg_BinA_only': 1.4529755115509033, 'fused_vs_marginal_gap': 1.654022455215454, 'gap_related': 1.654022455215454, 'gap_unrelated': nan}
    48900 {'loss': 0.06832772493362427, 'nll_fused_A': -0.28023433685302734, 'nll_fused_BinA': -0.2723730802536011, 'nll_marg_A': 1.4159036874771118, 'nll_marg_BinA_only': 1.411766529083252, 'fused_vs_marginal_gap': 1.684139609336853, 'gap_related': 1.6841394901275635, 'gap_unrelated': nan}
    49000 {'loss': 0.02854210138320923, 'nll_fused_A': -0.29860398173332214, 'nll_fused_BinA': -0.3012726902961731, 'nll_marg_A': 1.397986650466919, 'nll_marg_BinA_only': 1.3975090980529785, 'fused_vs_marginal_gap': 1.6987818479537964, 'gap_related': 1.6987817287445068, 'gap_unrelated': nan}
    49100 {'loss': 0.21095561981201172, 'nll_fused_A': -0.19820645451545715, 'nll_fused_BinA': -0.17279040813446045, 'nll_marg_A': 1.4773597717285156, 'nll_marg_BinA_only': 1.4818021059036255, 'fused_vs_marginal_gap': 1.654592514038086, 'gap_related': 1.654592514038086, 'gap_unrelated': nan}
    49200 {'loss': 0.13551342487335205, 'nll_fused_A': -0.2340705692768097, 'nll_fused_BinA': -0.22808319330215454, 'nll_marg_A': 1.446059226989746, 'nll_marg_BinA_only': 1.4586548805236816, 'fused_vs_marginal_gap': 1.6867380142211914, 'gap_related': 1.6867380142211914, 'gap_unrelated': nan}
    49300 {'loss': 0.08577311038970947, 'nll_fused_A': -0.25903433561325073, 'nll_fused_BinA': -0.27381521463394165, 'nll_marg_A': 1.4576621055603027, 'nll_marg_BinA_only': 1.4668446779251099, 'fused_vs_marginal_gap': 1.7406598329544067, 'gap_related': 1.7406599521636963, 'gap_unrelated': nan}
    49400 {'loss': 0.098195880651474, 'nll_fused_A': -0.24902932345867157, 'nll_fused_BinA': -0.25700831413269043, 'nll_marg_A': 1.4330432415008545, 'nll_marg_BinA_only': 1.4100165367126465, 'fused_vs_marginal_gap': 1.667024850845337, 'gap_related': 1.667024850845337, 'gap_unrelated': nan}
    49500 {'loss': 0.07172724604606628, 'nll_fused_A': -0.2798812687397003, 'nll_fused_BinA': -0.2856098711490631, 'nll_marg_A': 1.4710049629211426, 'nll_marg_BinA_only': 1.4614063501358032, 'fused_vs_marginal_gap': 1.747016191482544, 'gap_related': 1.747016191482544, 'gap_unrelated': nan}
"""

    # Option A: Plot a specific subset of metrics
    plot_learning_curves(
        log_output,
        metrics_to_plot=["loss", "nll_fused_A", "nll_marg_A"],
        title="Selected Losses",
    )

    # Option B: Plot all valid metrics automatically
    # plot_learning_curves(log_output)


