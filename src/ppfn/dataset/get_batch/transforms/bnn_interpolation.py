import numpy as np

from ppfn.dataset.get_batch.transforms.abstract_transform import TaskTransform
from ppfn.dataset.prior import MultiFidelityTask


class OutputInterpolationTransform(TaskTransform):
    """
    Creates a new task by linearly blending the outputs of the target task and a random task.

    Intent:
        To create a "smooth transition" or "task-mashing" effect, similar to
        MixUp augmentation in computer vision.

    Mechanism:
        Samples a completely new random task (Task B) and returns a weighted
        average of the target task (Task A) and Task B: y_new = (1-α)y_A + αy_B.

    Meta-Learning Intuition:
        Forces the meta-learner to handle "noisy" or "hybrid" tasks, improving
        robustness by populating the gaps between discrete points in the task prior.
    """

    def __init__(self, alpha=None, sample_alpha_fn=None, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax
        self.alpha = alpha
        self.sample_alpha_fn = sample_alpha_fn or (lambda: np.random.beta(1, 4))

    def plot_alpha_distribution(self, num_samples=1000):
        import matplotlib.pyplot as plt
        alphas = [self.sample_alpha_fn() for _ in range(num_samples)]
        plt.hist(alphas, bins=30, density=True)
        plt.title("Sampled Alpha Distribution")
        plt.xlabel("Alpha")
        plt.ylabel("Density")
        plt.show()

    def __call__(self, target_task: MultiFidelityTask):
        # 1. related_task is our "Task B"
        related_task = target_task.clone()
        y0, ymax = target_task.y0, target_task.ymax
        related_task.sample_task()

        if not self.resample_y0_ymax:
            related_task.y0 = y0
            related_task.ymax = ymax

        # 2. CAPTURE the original methods before overwriting
        # This is the key to preventing recursion!
        ancestor_get_curve_target = target_task.get_marginal_curve
        ancestor_get_curve_related = related_task.get_marginal_curve

        if self.alpha is None:
            alpha = self.sample_alpha_fn()
        else:
            alpha = self.alpha

        # 3. Overwrite with an interpolating wrapper
        def interpolated_get_marginal_curve(hyperparams):
            # Call the SAVED ancestor methods, NOT the instance methods
            curve_fn_a = ancestor_get_curve_target(hyperparams)
            curve_fn_b = ancestor_get_curve_related(hyperparams)

            def blended_curve(x, cid=0, noise=True):
                y_a = curve_fn_a(x, cid, noise=noise)
                y_b = curve_fn_b(x, cid, noise=noise)
                return (1 - alpha) * y_a + alpha * y_b

            return blended_curve

        # Patch the related_task instance
        related_task.get_marginal_curve = interpolated_get_marginal_curve
        return related_task, 1.0 - alpha

if __name__ == '__main__':

    from ppfn.dataset.get_batch.get_related_batch import get_batch

    # Example usage

    batch = get_batch(
        batch_size=4,
        seq_len=32,
        num_features=3,
        single_eval_pos=16,
        device="cpu",
        transform=OutputInterpolationTransform(alpha=0.3, resample_y0_ymax=False),
        share_unrelated=0.5,  # 50% of pairs will be unrelated
    )