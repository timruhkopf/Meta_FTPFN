from ppfn.prior import MultiFidelityTask
from ppfn.prior import BNNPrior
from ppfn.prior import VectorizedParameterLinker


# --- Core Composite Structure ---

class CompositeTask(MultiFidelityTask):
    def __init__(self, base_task: MultiFidelityTask):
        super().__init__(base_task.num_inputs, base_task.num_outputs)
        self.__dict__.update(base_task.clone().__dict__)

        # FIX 1: Explicitly create fresh lists to prevent state leakage across tasks
        self.input_ops = list(getattr(base_task, 'input_ops', []))
        self.fidelity_ops = list(getattr(base_task, 'fidelity_ops', []))
        self.output_ops = list(getattr(base_task, 'output_ops', []))

    def clone(self):
        # FIX 4: Ensure cloning a CompositeTask preserves the operation pipelines
        cloned = super().clone()
        if not isinstance(cloned, CompositeTask):
            cloned = CompositeTask(cloned)
        cloned.input_ops = list(self.input_ops)
        cloned.fidelity_ops = list(self.fidelity_ops)
        cloned.output_ops = list(self.output_ops)
        return cloned

    def get_marginal_curve(self, hyperparams):
        original_hps = hyperparams  # Save the un-warped/un-expanded HPs
        processed_hps = hyperparams

        for op in self.input_ops:
            processed_hps = op(processed_hps)

        base_curve_fn = super().get_marginal_curve(processed_hps)

        def pipeline_curve_fn(fidelities, cid=None, noise=True):
            t = np.asanyarray(fidelities)
            for op in self.fidelity_ops:
                t = op(t)

            y = base_curve_fn(t, cid=cid, noise=noise)

            for op in self.output_ops:
                # FIX 3: Pass original_hps down so other tasks don't get dimension mismatches
                y = op(y, t, original_hps, processed_hps, cid, noise)
            return y

        return pipeline_curve_fn


# --- Robust Operations ---
# These are designed to be modular and composable, and to avoid side effects on the underlying task structure.
# Each operation is a callable that transforms either the input hyperparameters, the fidelity levels, or the output curves.

class PowerWarpOp:
    def __init__(self, powers):
        self.powers = powers

    def __call__(self, x):
        d_warp = self.powers.shape[1]
        warped_part = torch.pow(x[:, :d_warp], self.powers.to(x.device))

        if x.shape[1] > d_warp:
            return torch.cat([warped_part, x[:, d_warp:]], dim=1)
        return warped_part


class FidelityWarpOp:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, t):
        return np.clip(np.power(t, self.alpha), 0, 1)


class LatentConcatOp:
    def __init__(self, val):
        self.val = val

    def __call__(self, x):
        ctx = torch.full((x.shape[0], 1), self.val, device=x.device, dtype=x.dtype)
        return torch.cat([x, ctx], dim=-1)


class InterpolationOp:
    def __init__(self, other_task, alpha):
        self.other_task = other_task
        self.alpha = alpha

    def __call__(self, y_a, t, original_hps, processed_hps, cid, noise):
        # FIX 3: other_task expects the base dimensions, so we feed it original_hps
        curve_b = self.other_task.get_marginal_curve(original_hps)
        y_b = curve_b(t, cid=cid, noise=noise)
        return (1 - self.alpha) * y_a + self.alpha * y_b


# --- Transforms ---


class TaskTransform:
    def __call__(self, target_task):
        raise NotImplementedError()

class SameTaskTransform(TaskTransform):
    """A no-op transform that returns the same task as both target and related, with perfect relatedness."""
    def __call__(self, task):
        return task, task.clone(), 1.0


class VectorizedInputWarpingTransform:
    def __init__(self, strength=0.4, p_drop=0.2):
        self.strength, self.p_drop = strength, p_drop

    def __call__(self, task):
        rel = CompositeTask(task)
        D = task.num_inputs
        mask = np.random.binomial(1, 1 - self.p_drop, size=D)
        powers = np.exp(mask * np.random.uniform(-self.strength, self.strength, size=D))
        rel.input_ops.append(PowerWarpOp(torch.tensor(powers, dtype=torch.float32).reshape(1, -1)))
        return rel, 1.0 - np.mean(np.abs(powers - 1))


class LatentInputTransform:
    def __call__(self, task: MultiFidelityTask):
        # FIX 2: Clone the incoming task BEFORE surgical modification!
        task_copy = task.clone()

        required_bnn_outputs = task_copy.bnn_prior.num_outputs
        latent_dim = task_copy.num_inputs + 1

        # it is a bit hacky to resample the mf_ftpfn_refactor that has the additional dimension,
        # but this allows us to keep the meta_batch logic as is.
        task_copy.bnn_prior = BNNPrior(latent_dim, required_bnn_outputs)
        task_copy.model = task_copy.bnn_prior.sample()
        task_copy.linker = VectorizedParameterLinker(task_copy.bnn_prior)

        l_target, l_related = np.random.uniform(0.0, 1), np.random.uniform(0,1 )

        target_comp = CompositeTask(task_copy)
        target_comp.input_ops.append(LatentConcatOp(l_target))

        related_comp = CompositeTask(task_copy)
        related_comp.input_ops.append(LatentConcatOp(l_related))

        return target_comp, related_comp, 1.0 - abs(l_target - l_related)


class OutputInterpolationTransform(TaskTransform):
    def __call__(self, task):
        rel = CompositeTask(task)
        task_b = task.clone()
        task_b.sample_task()
        alpha = np.random.beta(1, 4)
        rel.output_ops.append(InterpolationOp(task_b, alpha))
        return rel, 1.0 - alpha


if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt


    def visualize_transformation(transforms, num_inputs=1, bnn_outputs=23):
        """
        Applies a sequence of transforms to a base task and plots the comparison.

        Args:
            transforms: A single TaskTransform or a list of TaskTransforms.
            num_inputs: Number of HP dimensions.
            bnn_outputs: The required output width for the BNN (23).
        """
        # 1. Setup Base Task
        base = MultiFidelityTask(num_inputs, bnn_outputs)
        base.sample_task()

        # 2. Apply Transforms
        if not isinstance(transforms, list):
            transforms = [transforms]

        target = base
        related = base
        total_relatedness = 1.0

        for transform in transforms:
            # Check if the transform returns (target, related, score) or (related, score)
            result = transform(related)
            if len(result) == 3:
                target, related, score = result
            else:
                related, score = result
            total_relatedness *= score

        # 3. Plotting
        fig = plt.figure(figsize=(14, 6))

        # Plot Target
        ax1 = fig.add_subplot(121, projection='3d')
        target.plot_surface(ax=ax1, title=f"Target Task\n(Inputs: {target.num_inputs})")

        # Plot Related
        ax2 = fig.add_subplot(122, projection='3d')
        transform_names = " + ".join([type(t).__name__ for t in transforms])
        related.plot_surface(
            ax=ax2,
            title=f"Related: {transform_names}\nRel: {total_relatedness:.2f}"
        )

        plt.tight_layout()
        plt.show()

    # Test a single transform
    print("Testing Vectorized Warping...")
    visualize_transformation(VectorizedInputWarpingTransform(strength=1.5))

    # Test the Latent pipeline (this returns 3 values)
    print("Testing Latent Transform...")
    visualize_transformation(LatentInputTransform())

    # Test Chaining: Latent -> Warping -> Interpolation
    print("Testing Full Chain...")
    chain = [
        LatentInputTransform(),
        VectorizedInputWarpingTransform(strength=0.8),
        OutputInterpolationTransform()
    ]
    visualize_transformation(chain)