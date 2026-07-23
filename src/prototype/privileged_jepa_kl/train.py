import torch
import torch.nn.functional as F

from prototype.privileged_jepa_kl.prior import DomainShiftRegressionPrior
from prototype.privileged_jepa_kl.model import DomainJEPAPFN


def kl_teacher_student(logits_teach, logits_stud, label_smoothing=0.0):
    """KL(p_teach(stop-grad) || p_stud). Reverse KL from the student's
    perspective (mode-seeking) -- see the discussion of forward vs.
    reverse KL if the teacher turns out to be overconfident in practice."""
    p_teach = F.softmax(logits_teach.detach(), dim=-1)
    if label_smoothing > 0:
        k = p_teach.shape[-1]
        p_teach = p_teach * (1 - label_smoothing) + label_smoothing / k
    log_p_stud = F.log_softmax(logits_stud, dim=-1)
    log_p_teach = torch.log(p_teach.clamp_min(1e-8))
    kl = (p_teach * (log_p_teach - log_p_stud)).sum(-1)
    return kl.mean()


def train(
    steps=10000,
    batch_size=64,
    n_A_tr=20,
    n_A_test=10,
    n_B_tr=40,
    alpha=1.0,          # weight on L_KL
    beta=0.5,           # weight on direct student NLL floor
    b_in_a_dropout=0.3, # context dropout on the teacher's oracle set
    lr=3e-4,
    device="cpu",
    log_every=100,
):
    prior = DomainShiftRegressionPrior(d_z=2)
    model = DomainJEPAPFN(d_x=2, n_bins=32, y_range=(-4.0, 4.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        batch = prior.sample_batch(batch_size, n_A_tr, n_A_test, n_B_tr, device=device)
        out = model(batch, b_in_a_dropout=b_in_a_dropout, return_attn=(step % log_every == 0))

        y_bins = out["y_bins"]
        L_nll_teach = F.cross_entropy(out["logits_teach"].reshape(-1, model.n_bins), y_bins.reshape(-1))
        L_nll_stud = F.cross_entropy(out["logits_stud"].reshape(-1, model.n_bins), y_bins.reshape(-1))
        L_kl = kl_teacher_student(out["logits_teach"], out["logits_stud"])

        loss = L_nll_teach + alpha * L_kl + beta * L_nll_stud

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                p_teach = F.softmax(out["logits_teach"], dim=-1)
                ent_teach = -(p_teach * torch.log(p_teach.clamp_min(1e-8))).sum(-1).mean().item()
                acc_teach = (out["logits_teach"].argmax(-1) == y_bins).float().mean().item()
                acc_stud = (out["logits_stud"].argmax(-1) == y_bins).float().mean().item()
            msg = (
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"NLL_teach {L_nll_teach.item():.4f} | KL {L_kl.item():.4f} | "
                f"NLL_stud {L_nll_stud.item():.4f} | teach_ent {ent_teach:.3f} | "
                f"acc_teach {acc_teach:.3f} | acc_stud {acc_stud:.3f}"
            )
            if out["attn_stud"]:
                last_layer_attn = out["attn_stud"][-1]
                msg += f" | student last-layer attn [Z_A, Z_B] = {last_layer_attn}"
            print(msg)

    return model


if __name__ == "__main__":
    train()
