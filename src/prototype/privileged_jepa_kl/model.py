import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    """Permutation-invariant self-attention encoder over (x, y) context tokens.
    Domain identity is injected via a learned per-token embedding added
    before self-attention -- this is what lets a single shared encoder
    still distinguish which view (A or B) a token came from."""

    def __init__(self, d_x, d_model=64, n_heads=4, n_layers=3, dim_ff=128,
                 n_domains=2, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_x + 1, d_model)
        self.domain_embed = nn.Embedding(n_domains, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x, y, domain_id):
        # x: [B, n, d_x], y: [B, n]
        tok = self.in_proj(torch.cat([x, y.unsqueeze(-1)], dim=-1))
        tok = tok + self.domain_embed.weight[domain_id]
        return self.encoder(tok)  # [B, n, d_model]


class QueryEmbed(nn.Module):
    """Embeds a test point's x only (no y) into query space. This is the
    information barrier: the predictor never sees x_A_test through the
    context encoder, only through this lightweight, label-free embedding."""

    def __init__(self, d_x, d_model=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_x, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        return self.proj(x)  # [B, n_q, d_model]


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, need_weights=False):
        attn_out, attn_w = self.attn(
            q, kv, kv, need_weights=need_weights, average_attn_weights=True
        )
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q, attn_w


class Predictor(nn.Module):
    """
    N-layer JOINT cross-attention decoder stack (not a sequential
    refine-then-predict pipeline). Query = test-point embedding; context =
    concatenation of whatever token sets are passed in (e.g. [Z_A, Z_B]
    for the student, [Z_c] for the teacher). Every layer attends to all
    context groups in the SAME softmax, and residual connections carry the
    original query frame forward through every layer -- so attending
    heavily to one group in one layer cannot permanently overwrite
    information relevant to another group.
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=3, dim_ff=128, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            # fixme why cross attention only? -- query update.
            CrossAttnBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, query, context_list, return_attn=False):
        kv = torch.cat(context_list, dim=1)  # [B, sum_n, d_model]
        group_sizes = [c.shape[1] for c in context_list]
        q = query
        attn_logs = []
        for block in self.blocks:
            q, attn_w = block(q, kv, need_weights=return_attn)
            if return_attn and attn_w is not None:
                # attn_w: [B, n_q, sum_n] -> average attention mass per context group
                splits = torch.split(attn_w, group_sizes, dim=-1)
                attn_logs.append([s.sum(-1).mean().item() for s in splits])
        return q, attn_logs


class BinnedHead(nn.Module):
    """Shared decode head: predictor output vector -> logits over fixed
    y-bins. Shared between teacher and student branches so the KL term
    compares distributions over an identical support."""

    def __init__(self, d_model, n_bins, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, n_bins))

    def forward(self, z):
        return self.net(z)


class DomainJEPAPFN(nn.Module):
    def __init__(self, d_x=2, d_model=64, n_heads=4, enc_layers=3, pred_layers=3,
                 dim_ff=128, n_bins=32, y_range=(-4.0, 4.0), n_domains=2, dropout=0.0):
        super().__init__()
        self.encoder = SetEncoder(d_x, d_model, n_heads, enc_layers, dim_ff, n_domains, dropout)
        self.query_embed = QueryEmbed(d_x, d_model)
        self.predictor = Predictor(d_model, n_heads, pred_layers, dim_ff, dropout)
        self.head = BinnedHead(d_model, n_bins)
        self.register_buffer("bin_edges", torch.linspace(y_range[0], y_range[1], n_bins + 1))
        self.n_bins = n_bins

    def bin_targets(self, y):
        idx = torch.bucketize(y, self.bin_edges[1:-1])
        return idx.clamp(0, self.n_bins - 1)

    def forward(self, batch, b_in_a_dropout=0.0, return_attn=False):
        x_A_tr, y_A_tr = batch["A_tr"]
        x_A_test, y_A_test = batch["A_test"]
        x_B_tr, y_B_tr = batch["B_tr"]
        x_BinA_tr, y_BinA_tr = batch["BinA_tr"]

        # --- student: two separate, unfused context encodings ---
        Z_A = self.encoder(x_A_tr, y_A_tr, domain_id=0)
        Z_B = self.encoder(x_B_tr, y_B_tr, domain_id=1)

        # --- teacher: single-domain oracle context (A_tr + B_in_A_tr) ---
        if b_in_a_dropout > 0.0 and self.training:
            n = x_BinA_tr.shape[1]
            # FIXME: drop b_in_a_dropout? ; the idea of it is to make the teacher more noisy, not penalizing
            #  the student as much for less sharp predictions
            keep = max(1, int(n * (1 - b_in_a_dropout)))
            idx = torch.randperm(n, device=x_BinA_tr.device)[:keep]
            x_BinA_used, y_BinA_used = x_BinA_tr[:, idx], y_BinA_tr[:, idx]
        else:
            x_BinA_used, y_BinA_used = x_BinA_tr, y_BinA_tr

        Z_c = self.encoder(
            torch.cat([x_A_tr, x_BinA_used], dim=1),
            torch.cat([y_A_tr, y_BinA_used], dim=1),
            domain_id=0,
        )

        query = self.query_embed(x_A_test)

        Z_hat_stud, attn_stud = self.predictor(query, [Z_A, Z_B], return_attn=return_attn)
        # fixme: shoudl the teacher not take the raw query on x_A_test pe?
        Z_hat_teach, attn_teach = self.predictor(query, [Z_c], return_attn=return_attn)

        logits_stud = self.head(Z_hat_stud)
        logits_teach = self.head(Z_hat_teach)

        y_bins = self.bin_targets(y_A_test)

        return {
            "logits_stud": logits_stud,
            "logits_teach": logits_teach,
            "y_bins": y_bins,
            "attn_stud": attn_stud,   # list per layer of [mass_on_Z_A, mass_on_Z_B]
            "attn_teach": attn_teach,
        }
