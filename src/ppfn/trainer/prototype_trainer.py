import logging

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from ppfn.trainer import PPFNTrainer

logger = logging.getLogger(__name__)


class TriStreamTrainer(PPFNTrainer):
    def __init__(self, warmup_steps, train_jointly, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps
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
            if self.global_step == 0:
                logger.info("Freezing Adapter C for warmup phase.")
                for param in self.model.layer.parameters():
                    param.requires_grad = False

                # now change the criterion state to change the objective as well!
                self.criterion.is_warmup = True

            elif self.global_step == self.warmup_steps:
                logger.info("Warmup complete. Freezing marginal backend; unlocking Adapter C.")
                for param in set(self.model.parameters()) - set(self.model.layer.parameters()):
                    param.requires_grad = False
                for param in self.model.layer.parameters():
                    param.requires_grad = True

                # now change the criterion state to change the objective as well!
                self.criterion.is_warmup = False

        return super()._train_step(step, batch, **fwd_kwargs)