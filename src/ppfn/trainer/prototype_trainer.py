import logging

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from ppfn.trainer import PPFNTrainer
from prototype.meta_jepa_v2.loss import  update_ema_dict_modules

logger = logging.getLogger(__name__)


class TriStreamTrainer(PPFNTrainer):
    def __init__(self, warmup_epochs, train_jointly, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.train_jointly = train_jointly
        self.criterion.train_jointly = self.train_jointly


    def _get_next_batch(self):
        batch = next(self.train_iter)

        # collate is identity, and batch is generated in the dataset class.
        # This is the PFN style :/
        batch = batch[0]

        # 3. Move nested dict batch to device
        for k1 in ['train', 'test', 'params']:
            if k1 in batch:
                for k2 in batch[k1]:
                    batch[k1][k2] = batch[k1][k2].to(self.device)
        return batch, {}

    def _set_active_parameters(self, active_param_ids: set[int]):
        """
        Elegant freezing utility. Disables gradients for all parameters in the model
        EXCEPT those whose memory addresses are explicitly in the active_param_ids set.
        """
        for param in self.model.parameters():
            # id(param) returns an int. We check if that int is in our set of active ints.
            param.requires_grad = id(param) in active_param_ids

    def _train_step(self, step, batch, **fwd_kwargs) -> dict[str, float]:
        # 1. Clear global context
        ForwardMetaContext.clear()

        # 2. Dynamic Freezing Logic (Stays the same, but using the cleaner .warmup_parameters_ids)
        if not self.train_jointly:
            transition_step = self.warmup_epochs * self.steps
            if self.global_step == 0:
                self.criterion.is_warmup = True
                self.model.lambda_jepa = 0.0
                self._set_active_parameters(self.model.warmup_parameters_ids)
            elif self.global_step == transition_step:
                self.criterion.is_warmup = False
                self.model.lambda_jepa = 1.0
                self._set_active_parameters(self.model.post_warmup_parameters_ids)

        # 3. Standard Forward / Backward / Optimizer Step
        out = super()._train_step(step, batch, **fwd_kwargs)

        # 4. Refactored EMA Updates
        if self.train_jointly or self.criterion.is_warmup:
            # We iterate through the student modules and map them to the teacher
            for key in self.model.student:
                # We skip embedding_norm if you want to handle it separately or
                # keep it in the loop if you want it tracked via EMA
                update_ema_dict_modules(
                    student_module=self.model.student[key],
                    teacher_module=self.model.teacher[key],
                    momentum=0.996
                )

        return out