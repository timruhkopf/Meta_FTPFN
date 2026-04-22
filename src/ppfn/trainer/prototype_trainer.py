import logging

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from ppfn.trainer import PPFNTrainer

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

    def _train_step(self, step, batch, **fwd_kwargs) -> dict[str, float]:
        # 1. Clear global context
        ForwardMetaContext.clear()

        # 2. Dynamic Freezing Logic
        if not self.train_jointly:

            # Calculate the exact step where warmup ends
            # (Assuming self.steps is 'steps_per_epoch')
            transition_step = self.warmup_epochs * self.steps

            if self.global_step == 0:
                logger.info("Phase 1: Freezing Adapter C for warmup phase.")
                for param in self.model.layer.parameters():
                    param.requires_grad = False

                self.criterion.is_warmup = True

            elif self.global_step == transition_step:
                logger.info("Phase 2: Warmup complete. Freezing marginal backend; unlocking Adapter C.")

                # Cleaner PyTorch idiom: Freeze everything first...
                for param in self.model.parameters():
                    param.requires_grad = False

                # ...then explicitly unfreeze the layer you want to train
                for param in self.model.layer.parameters():
                    param.requires_grad = True

                self.criterion.is_warmup = False
        return super()._train_step(step, batch, **fwd_kwargs)