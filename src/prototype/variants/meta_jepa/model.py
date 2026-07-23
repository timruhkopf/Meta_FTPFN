import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from prototype.meta_jepa.loss import compute_jepa_loss
from prototype.variants.meta_jepa.pfn import PFNStack


class MetaJEPAPFN(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, embed_dim: int, num_heads: int,
                 enc_layers: int, pred_layers: int, pfn_layers: int, num_bars: int,
                 lambda_jepa: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_jepa = lambda_jepa

        # Input Embeddings
        self.E_x = nn.Linear(x_dim, embed_dim)
        self.E_y = nn.Linear(y_dim, embed_dim)

        # Domain Signatures
        self.e_target = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.e_source = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Network Stacks
        self.student_encoder_stack = PFNStack(embed_dim, enc_layers, num_heads)
        self.predictor_stack = PFNStack(embed_dim, pred_layers, num_heads)
        self.final_pfn_stack = PFNStack(embed_dim, pfn_layers, num_heads)

        # Loss Criterion and Prediction Head

        self.output_dim = num_bars
        self.prediction_head = nn.Linear(embed_dim, self.output_dim)

        # Freeze Teacher
        self.teacher_encoder_stack = copy.deepcopy(self.student_encoder_stack)
        for param in self.teacher_encoder_stack.parameters():
            param.requires_grad = False

    def _embed(self, x: Tensor, y: Optional[Tensor], domain: str) -> Tensor:
        emb = self.E_x(x)
        if y is not None:
            emb = emb + self.E_y(y)

        return emb
        domain_emb = self.e_target if domain == 'target' else self.e_source
        return emb + domain_emb

    def forward(self, prior_data: dict):

        train = prior_data['train']
        test = prior_data['test']

        mask_A = train['padding_mask_A']
        mask_B = train['padding_mask_B']
        mask_Joint = torch.cat([mask_A, mask_B], dim=1) if mask_A is not None and mask_B is not None else None

        # ==========================================
        # 1. TEACHER PASS (EMA Target)
        # ==========================================
        with torch.no_grad():
            BinA_emb = self._embed(train['X_B_in_A'], train['Y_B_in_A'], domain='target')
            Z_teacher_BinA, _ = self.teacher_encoder_stack(context=BinA_emb, queries=None, pad_mask=mask_B)

        # ==========================================
        # 2. STUDENT ENCODER (A, B, and Q_A)
        # ==========================================
        A_emb = self._embed(train['X_A'], train['Y_A'], domain='target')
        Q_A_emb = self._embed(test['X_A'], y=None, domain='target')
        B_emb = self._embed(train['X_B'], train['Y_B'], domain='source')
        Q_B_emb = self._embed(test['X_B'], y=None, domain='source')

        Z_A, z_QA = self.student_encoder_stack(context=A_emb, queries=Q_A_emb, pad_mask=mask_A)
        Z_B, z_QB = self.student_encoder_stack(context=B_emb, queries=Q_B_emb, pad_mask=mask_B)

        ForwardMetaContext.set(
            kwargs={"Telemetry/Latent_Variance_A": Z_A.var(dim=(0, 1)).mean().item(),
                    "Telemetry/Latent_Variance_B": Z_B.var(dim=(0, 1)).mean().item()}
        )

        # ==========================================
        # 3. JEPA PREDICTOR
        # ==========================================
        # Z_B queries Z_A to perform the alignment
        _, Z_hat_B = self.predictor_stack(context=Z_A, queries=Z_B, pad_mask=mask_A)

        jepa_loss = compute_jepa_loss(Z_hat_B, Z_teacher_BinA.detach(), mode='cosine')
        ForwardMetaContext.set("Telemetry/JEPA_Cosine_Loss", jepa_loss.item())
        ForwardMetaContext.set("aux_loss/jepa", jepa_loss * self.lambda_jepa)

        # ==========================================
        # 4. FINAL INFERENCE (Joint vs A-Only)
        # ==========================================
        Joint_Context = torch.cat([Z_A, Z_hat_B], dim=0)

        # Production Pass
        _, final_latent_Q_joint = self.final_pfn_stack(context=Joint_Context, queries=z_QA, pad_mask=mask_Joint)
        logits_joint = self.prediction_head(final_latent_Q_joint)

        # Marginal Telemetry Pass
        # with torch.no_grad():
        _, final_latent_Q_A = self.final_pfn_stack(context=Z_A, queries=z_QA, pad_mask=mask_A)
        logits_A = self.prediction_head(final_latent_Q_A)

        _, final_latent_Q_B = self.final_pfn_stack(context=Z_B, queries=z_QB, pad_mask=mask_B)
        logits_B = self.prediction_head(final_latent_Q_B)

        # ==========================================
        # 5. TELEMETRY & DISTRIBUTION LOSS
        # ==========================================
        probs_joint = F.softmax(logits_joint, dim=-1)
        entropy_joint = -(probs_joint * torch.log(probs_joint + 1e-9)).sum(dim=-1).mean().item()

        probs_A = F.softmax(logits_A, dim=-1)
        entropy_A = -(probs_A * torch.log(probs_A + 1e-9)).sum(dim=-1).mean().item()

        ForwardMetaContext.set(
            kwargs={"Telemetry/PPD_Entropy_Joint": entropy_joint,
                    "Telemetry/PPD_Entropy_A": entropy_A,
                    }
        )

        return logits_A, logits_B, logits_joint

    def forward_inference(self, A_x: Tensor, A_y: Tensor, B_x: Tensor, B_y: Tensor, Q_x: Tensor,
                          mask_A: Optional[Tensor] = None, mask_B: Optional[Tensor] = None):
        """Standard zero-shot inference without teacher or loss calculations."""
        mask_Joint = torch.cat([mask_A, mask_B], dim=1) if mask_A is not None and mask_B is not None else None

        A_emb = self._embed(A_x, A_y, domain='target')
        Q_emb = self._embed(Q_x, y=None, domain='target')
        B_emb = self._embed(B_x, B_y, domain='source')

        Z_A, z_Q = self.student_encoder_stack(context=A_emb, queries=Q_emb, pad_mask=mask_A)
        Z_B, _ = self.student_encoder_stack(context=B_emb, queries=None, pad_mask=mask_B)

        _, Z_hat_B = self.predictor_stack(context=Z_A, queries=Z_B, pad_mask=mask_A)

        Joint_Context = torch.cat([Z_A, Z_hat_B], dim=0)
        _, final_latent_Q = self.final_pfn_stack(context=Joint_Context, queries=z_Q, pad_mask=mask_Joint)

        return self.prediction_head(final_latent_Q)
