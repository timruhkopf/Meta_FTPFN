import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from prototype.meta_jepa_v2.pfn import PFNStack


class MetaJEPAPFN(nn.Module): # ANAMORPHISM
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
        self.predictor_align_stack = PFNStack(embed_dim, pred_layers, num_heads)
        self.predictor_infer_stack = PFNStack(embed_dim, pfn_layers, num_heads)

        # Shared Decoder Head

        self.output_dim = num_bars
        self.prediction_head = nn.Linear(embed_dim, self.output_dim)

        # Teacher (EMA Oracle)
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

    def forward(self, prior_data: dict, is_warmup: bool = False, train_jointly: bool = False):
        ForwardMetaContext.clear()

        train = prior_data['train']
        test = prior_data['test']

        mask_A = train['padding_mask_A']
        mask_B = train['padding_mask_B']
        mask_Joint = torch.cat([mask_A, mask_B], dim=1) if mask_A is not None and mask_B is not None else None

        # ==========================================
        # 1. TEACHER PASS (EMA Oracle)
        # ==========================================
        with torch.no_grad():
            # Oracle gets perfect B_in_A and uses it to answer Q_A
            BinA_emb = self._embed(train['X_B_in_A'], train['Y_B_in_A'], domain='target')
            Q_A_emb_teacher = self._embed(test['X_A'], y=None, domain='target')

            _, Z_teacher_QA = self.teacher_encoder_stack(context=BinA_emb, queries=Q_A_emb_teacher, pad_mask=mask_B)
            logits_teacher_QA = self.prediction_head(Z_teacher_QA)

        # ==========================================
        # 2. STUDENT ENCODER (Marginals)
        # ==========================================
        A_emb = self._embed(train['X_A'], train['Y_A'], domain='target')
        Q_A_emb = self._embed(test['X_A'], y=None, domain='target')

        B_emb = self._embed(train['X_B'], train['Y_B'], domain='source')
        Q_B_emb = self._embed(test['X_B'], y=None, domain='source')

        Z_A, Z_QA_A = self.student_encoder_stack(context=A_emb, queries=Q_A_emb, pad_mask=mask_A)
        Z_B, Z_QB_B = self.student_encoder_stack(context=B_emb, queries=Q_B_emb, pad_mask=mask_B)

        ForwardMetaContext.set(
            kwargs={"Telemetry/Latent_Variance_A": Z_A.var(dim=(0, 1)).mean().item(),
                    "Telemetry/Latent_Variance_B": Z_B.var(dim=(0, 1)).mean().item()}
        )

        # ==========================================
        # 3. PREDICTOR (Align & Infer)
        # ==========================================
        # THE DETACH SHIELD: If we are ONLY training on C, we must protect the encoder
        # from the "random mirror" effect of the untrained predictor.
        # if not train_jointly and not is_warmup:
        #     Z_A_for_align = Z_A.detach()
        #     Z_B_for_align = Z_B.detach()
        # else:
        Z_A_for_align = Z_A
        Z_B_for_align = Z_B

        # Phase 1: Align (B queries A)
        _, Z_hat_B = self.predictor_align_stack(context=Z_A_for_align, queries=Z_B_for_align, pad_mask=mask_A)

        # Phase 2: Infer (Q_A queries Joint Context)
        Joint_Context = torch.cat([Z_A_for_align, Z_hat_B], dim=0)
        _, Z_QA_C = self.predictor_infer_stack(context=Joint_Context, queries=Z_QA_A, pad_mask=mask_Joint)

        # ==========================================
        # 4. SHARED DECODER HEAD
        # ==========================================
        logits_A = self.prediction_head(Z_QA_A)
        logits_B = self.prediction_head(Z_QB_B)
        logits_C = self.prediction_head(Z_QA_C)

        # ==========================================
        # 5. JEPA KNOWLEDGE DISTILLATION
        # ==========================================
        # Reshape for KL Div: (Time * Batch, Num_Bars)
        student_log_probs = F.log_softmax(logits_C, dim=-1).view(-1, self.output_dim)
        teacher_probs = F.softmax(logits_teacher_QA.detach(), dim=-1).view(-1, self.output_dim)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        ForwardMetaContext.set("aux_loss/jepa", kl_loss * self.lambda_jepa)
        ForwardMetaContext.set("Telemetry/JEPA_KL_Loss", kl_loss.item())

        # ==========================================
        # 6. DISTRIBUTIONAL TELEMETRY
        # ==========================================
        probs_C = F.softmax(logits_C, dim=-1)
        entropy_C = -(probs_C * torch.log(probs_C + 1e-9)).sum(dim=-1).mean().item()

        probs_A = F.softmax(logits_A, dim=-1)
        entropy_A = -(probs_A * torch.log(probs_A + 1e-9)).sum(dim=-1).mean().item()

        ForwardMetaContext.set(
            kwargs={"Telemetry/PPD_Entropy_Joint": entropy_C,
                    "Telemetry/PPD_Entropy_A": entropy_A}
        )

        return logits_A, logits_B, logits_C