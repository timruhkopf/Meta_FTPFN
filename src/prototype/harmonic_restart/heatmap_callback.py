import os
import matplotlib.pyplot as plt
import mlflow

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
from prototype.harmonic_restart.harmonic_prior import HarmonicsVisualizer


class HeatmapCallback(AbstractCallback):  # Assuming you inherit from your AbstractCallback
    def __init__(self, plot_every: int, plot_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def on_epoch_end(self, **kwargs):

        step = self.trainer.global_step


        if step % self.plot_every == 0:
            batch, _ = self.trainer._get_next_batch()

            logits_A, logits_B, logits_C = self.trainer.model(batch)

            fig = plt.figure(figsize=(10, 8))
            plot_name = f"heatmaps_step_{step:05d}.png"
            plot_path = os.path.join(self.plot_dir, plot_name)



            # Updated to use the separated Visualizer class
            HarmonicsVisualizer.save_heatmaps(
                fig=fig,
                batch_data=batch,
                borders=self.trainer.criterion.criterion_backend.borders,
                save_path=plot_path,
                logits_A=logits_A,
                logits_B=logits_B,
                logits_C=logits_C,
                plot=False
            )

            mlflow.log_artifact(plot_path, "heatmap_plots")

