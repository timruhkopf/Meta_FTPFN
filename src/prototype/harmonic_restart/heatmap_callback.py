import os
import matplotlib.pyplot as plt
import mlflow

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
from prototype.harmonic_restart.harmonic_prior import HeatmapVisualizer


class HeatmapCallback(AbstractCallback):  # Assuming you inherit from your AbstractCallback
    def __init__(self, plot_every: int, plot_dir: str, start_plotting_epoch=2000, **kwargs):
        super().__init__(**kwargs)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        self.start_plotting = start_plotting_epoch
        os.makedirs(self.plot_dir, exist_ok=True)

    def on_epoch_end(self, epoch, **kwargs):
        if epoch >= self.start_plotting and epoch % self.plot_every == 0:
            batch, _ = self.trainer._get_next_batch()

            logits_A, logits_B, logits_C = self.trainer.model(batch)

            fig = plt.figure(figsize=(10, 8))
            plot_name = f"heatmaps_step_{epoch:05d}.png"
            plot_path = os.path.join(self.plot_dir, plot_name)



            # Updated to use the separated Visualizer class
            HeatmapVisualizer.save_heatmaps(
                fig=fig,
                batch_data=batch,
                borders=self.trainer.criterion.criterion_backend.borders,
                save_path=plot_path,
                model=self.trainer.model,
                # logits_A=logits_A,
                # logits_B=logits_B,
                # logits_C=logits_C,
                plot=False
            )

            mlflow.log_artifact(plot_path, "heatmap_plots")

