"""Training objective.

    L = CE_bar(A_test)                       the actual task
      + w_kl  * softCE(teacher || student)   privileged-teacher distillation
      + w_y   * CE_bar(B_hat.y -> B_inA.y)   y-only translation (LUPI)
      + w_var * anti_collapse(B_hat)         stops B_hat folding onto A_train
      + w_sev * |sigmoid(s_hat) - s|         relatedness supervision
      + w_mmd * MMD(B_hat, A_pool)           optional, unpaired, no OT needed

Every term except the first uses information the sampler knows and the model
must not see at test time.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution


@dataclasses.dataclass
class LossWeights:
    kl: float = 1.0
    translation: float = 0.5
    variance: float = 0.1
    severity: float = 0.1
    mmd: float = 0.0


def bar_nll(
    criterion: FullSupportBarDistribution,
    logits_TBK: torch.Tensor,
    y_TB: torch.Tensor,
) -> torch.Tensor:
    """Negative log density under the bar distribution. Shapes are TabPFN's."""
    return criterion(logits_TBK, y_TB).mean()


def teacher_soft_ce(
    criterion: FullSupportBarDistribution,
    student_logits_TBK: torch.Tensor,
    teacher_logits_TBK: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft cross-entropy against the teacher's bar distribution.

    Uses ``criterion.full_ce``, so this is KL(teacher || student) up to the
    teacher's entropy, which is constant w.r.t. the student.

    Only valid when both models share ``borders`` -- hence the fixed grid in
    ``model.make_borders`` rather than A_train quantiles.
    """
    with torch.no_grad():
        teacher_probs = torch.softmax(teacher_logits_TBK / temperature, dim=-1)
    return criterion.full_ce(student_logits_TBK / temperature, teacher_probs).mean()


def anti_collapse(z_BRE: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg variance + covariance penalty on B_hat's row embeddings.

    This guards the dominant failure mode: the aligner is grounded on a sparse
    A_train, so the cheap solution is to map all of B onto A_train's support.
    Training NLL looks fine, attention looks healthy, and the extra shape
    information that motivated the whole design is gone. Track effective rank
    alongside this.
    """
    z = z_BRE - z_BRE.mean(dim=1, keepdim=True)
    std = torch.sqrt(z.var(dim=1) + 1e-6)  # (B, E)
    var_loss = F.relu(gamma - std).mean()

    B, R, E = z.shape
    cov = torch.einsum("bre,brf->bef", z, z) / max(R - 1, 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    cov_loss = off_diag.pow(2).sum(dim=(-2, -1)).mean() / E
    return var_loss + cov_loss


@torch.no_grad()
def effective_rank(z_BRE: torch.Tensor) -> torch.Tensor:
    """exp(entropy of the normalised singular value spectrum). Diagnostic only."""
    z = z_BRE - z_BRE.mean(dim=1, keepdim=True)
    s = torch.linalg.svdvals(z.float())
    p = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(p * (p + 1e-12).log()).sum(dim=-1)).mean()


def energy_distance(x_BRE: torch.Tensor, y_BSE: torch.Tensor) -> torch.Tensor:
    """Sample-based set distance. Handles |B| != |A_pool| natively.

    This is the reason the OT framing is unnecessary: attention imposes no
    marginal constraint, and where we *do* want a distributional objective, an
    energy distance needs no coupling to solve and no mass-balance assumption.
    """

    def pdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cdist(a, b).mean(dim=(-2, -1))

    return (2 * pdist(x_BRE, y_BSE) - pdist(x_BRE, x_BRE) - pdist(y_BSE, y_BSE)).mean()


def compute_loss(
    *,
    criterion: FullSupportBarDistribution,
    student_logits_MBK: torch.Tensor,
    y_A_test_MB: torch.Tensor,
    aux,  # model.AuxOutputs | None
    weights: LossWeights,
    teacher_logits_MBK: torch.Tensor | None = None,
    y_B_inA_RB: torch.Tensor | None = None,
    severity_B: torch.Tensor | None = None,
    a_pool_rows_BSE: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    parts: dict[str, torch.Tensor] = {
        "nll": bar_nll(criterion, student_logits_MBK, y_A_test_MB)
    }

    if teacher_logits_MBK is not None and weights.kl > 0:
        parts["kl"] = weights.kl * teacher_soft_ce(
            criterion, student_logits_MBK, teacher_logits_MBK
        )

    if aux is not None:
        if y_B_inA_RB is not None and weights.translation > 0:
            parts["translation"] = weights.translation * bar_nll(
                criterion, aux.b_hat_y_logits_RBK, y_B_inA_RB
            )
        if weights.variance > 0:
            parts["variance"] = weights.variance * anti_collapse(aux.b_hat_rows_BRE)
        if severity_B is not None and weights.severity > 0:
            pred = torch.sigmoid(aux.severity_logit_B1).squeeze(-1)
            parts["severity"] = weights.severity * (pred - severity_B).abs().mean()
        if a_pool_rows_BSE is not None and weights.mmd > 0:
            parts["mmd"] = weights.mmd * energy_distance(
                aux.b_hat_rows_BRE, a_pool_rows_BSE
            )

    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}
