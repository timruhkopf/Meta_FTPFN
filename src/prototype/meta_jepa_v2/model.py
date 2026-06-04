import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from prototype.meta_jepa_v2.pfn import PFNStack


class MetaJEPAPFN(nn.Module):  # ANAMORPHISM
    def __init__(self, x_dim: int, y_dim: int, embed_dim: int, num_heads: int,
                 enc_layers: int, pred_layers: int, pfn_layers: int, num_bars: int,
                 lambda_jepa: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_jepa = lambda_jepa

        # Domain Signatures
        self.e_target = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.e_source = nn.Parameter(torch.randn(1, 1, embed_dim))


        self.predictor_align_stack = PFNStack(embed_dim, pred_layers, num_heads)  # B to A cross attn
        self.predictor_infer_stack = PFNStack(embed_dim, pfn_layers, num_heads)  # QA to [A, hat B_in_A] cross attn

        self.embedding_norm = nn.LayerNorm(embed_dim)        # Shared Decoder Head

        self.output_dim = num_bars

        # Student Registry
        self.student = nn.ModuleDict({
            'E_x': nn.Linear(x_dim, embed_dim),
            'E_y': nn.Linear(y_dim, embed_dim),
            'embedding_norm': nn.LayerNorm(embed_dim),
            'encoder_stack': PFNStack(embed_dim, enc_layers, num_heads),
            'prediction_head': nn.Linear(embed_dim, self.output_dim)
        })

        # Teacher Registry (mirrored)
        self.teacher = nn.ModuleDict({
            k: copy.deepcopy(v) for k, v in self.student.items()
        })
        for param in self.teacher.parameters():
            param.requires_grad = False

    @property
    def warmup_parameters_ids(self) -> set[int]:
        """Phase 1: All student parameters are warmup parameters."""
        return {id(p) for p in self.student.parameters()}

    @property
    def teacher_parameters_ids(self) -> set[int]:
        return {id(p) for p in self.teacher.parameters()}

    @property
    def post_warmup_parameters_ids(self) -> set[int]:
        """
        Phase 2 parameters: The Alignment.
        Dynamically extracts ALL model parameters except the Teacher's parameters
        and the Phase 1 Warmup parameters.
        """
        # Pure integer set subtraction - lightning fast and perfectly safe!
        return {id(p) for p in self.parameters()} - self.teacher_parameters_ids - self.warmup_parameters_ids

    def _embed(self, x: Tensor, y: Optional[Tensor], registry: nn.ModuleDict) -> Tensor:
        emb = registry['E_x'](x)
        if y is not None:
            emb = emb + registry['E_y'](y)

        return registry['embedding_norm'](emb)

    def forward(self, prior_data: dict, is_warmup: bool = False, train_jointly: bool = False):
        ForwardMetaContext.clear()
        train, test = prior_data['train'], prior_data['test']
        mask_A, mask_B = train['padding_mask_A'], train['padding_mask_B']
        mask_Joint = torch.cat([mask_A, mask_B], dim=1) if mask_A is not None else None

        # 1. TEACHER PASS (EMA Oracle)
        with torch.no_grad():
            A_emb_teacher = self._embed(train['X_A'], train['Y_A'], self.teacher)
            BinA_emb = self._embed(train['X_B_in_A'], train['Y_B_in_A'], self.teacher)
            Joint_Oracle_Context = torch.cat([A_emb_teacher, BinA_emb], dim=0)

            Q_A_emb_teacher = self._embed(test['X_A'], None, self.teacher)

            _, Z_teacher_QA = self.teacher['encoder_stack'](
                context=Joint_Oracle_Context, queries=Q_A_emb_teacher, pad_mask=mask_Joint
            )
            logits_teacher_QA = self.teacher['prediction_head'](Z_teacher_QA)

        # 2. STUDENT ENCODER (Marginals)
        A_emb = self._embed(train['X_A'], train['Y_A'], self.student)
        Q_A_emb = self._embed(test['X_A'], None, self.student)
        B_emb = self._embed(train['X_B'], train['Y_B'], self.student)
        Q_B_emb = self._embed(test['X_B'], None, self.student)

        Z_A, Z_QA_A = self.student['encoder_stack'](context=A_emb, queries=Q_A_emb, pad_mask=mask_A)
        Z_B, Z_QB_B = self.student['encoder_stack'](context=B_emb, queries=Q_B_emb, pad_mask=mask_B)

        # 3. PREDICTOR (Align & Infer)
        _, Z_hat_B = self.predictor_align_stack(context=Z_A, queries=Z_B, pad_mask=mask_A)

        Joint_Context = torch.cat([Z_A, Z_hat_B], dim=0)
        _, Z_QA_C = self.predictor_infer_stack(context=Joint_Context, queries=Z_QA_A, pad_mask=mask_Joint)

        # 4. DECODER HEAD
        logits_A = self.student['prediction_head'](Z_QA_A)
        logits_B = self.student['prediction_head'](Z_QB_B)
        logits_C = self.student['prediction_head'](Z_QA_C)

        # 5. DISTILLATION
        with torch.amp.autocast('cuda', enabled=False):
            T = 2.0
            student_log_probs = F.log_softmax(logits_C.float() / T, dim=-1).view(-1, self.output_dim)
            teacher_probs = F.softmax(logits_teacher_QA.detach().float() / T, dim=-1).view(-1, self.output_dim)

            soft_ce_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean() * (T * T)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

        ForwardMetaContext.set("aux_loss/jepa", soft_ce_loss * self.lambda_jepa)
        ForwardMetaContext.set("Telemetry/JEPA_KL_Loss", kl_loss.item())
        ForwardMetaContext.set("Telemetry/JEPA_Soft_CE_Loss", soft_ce_loss.item())
        ForwardMetaContext.set("logits_teacher_QA", logits_teacher_QA)

        return logits_A, logits_B, logits_C