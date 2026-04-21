import os
import matplotlib.pyplot as plt
import mlflow
import torch
from ppfn.trainer.callbacks.abstract_callback import AbstractCallback


class HeatmapCallback(AbstractCallback):  # Assuming you inherit from your AbstractCallback
    def __init__(self, plot_every: int, plot_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def on_epoch_end(self, **kwargs):

        step = self.trainer.global_step

        batch, _ = self.trainer._get_next_batch()

        if step % self.plot_every == 0:
            fig = plt.figure(figsize=(10, 8))
            plot_name = f"heatmaps_step_{step:05d}.png"
            plot_path = os.path.join(self.plot_dir, plot_name)

            # Assuming trainer.model is accessible
            self.trainer.train_loader.dataset.save_heatmaps(
                fig=fig,
                batch_data=batch,  # Make sure batch is accessible here
                borders=self.trainer.criterion.criterion_backend.borders,
                save_path=plot_path,
                model=self.trainer.model
            )
            mlflow.log_artifact(plot_path, "heatmap_plots")

