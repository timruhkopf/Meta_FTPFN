import torch 
import numpy as np 

class AllocationPrior:
    EPS = 10**-9

    def __init__(self, seq_len, n_levels) -> None:
        """Determine the allocation of observations/queries per curve in the sequence."""
        self.seq_len = seq_len
        self.n_levels = n_levels
        self.sample_hp()

    def sample_hp(self):
        """Dirichlet-like sampling of allocation weights over the curves in the sequence."""
        self.alpha = 10 ** np.random.uniform(-4, -1)
        self.weights = np.random.gamma(self.alpha, self.alpha, self.seq_len) + self.EPS
        self.p = self.weights / np.sum(self.weights)

    def sample_abstract_allocation(self, single_eval_pos):
        """
        Sample observation and query positions for multiple curves based on a probabilistic ordering.
        This method determines how many observations (cutoff) and total epochs are allocated to each
        curve in the sequence. It uses a weighted sampling approach to assign positions to curves,
        then counts observations and epochs for each curve based on a cutoff position.
        Args:
            single_eval_pos (int): The cutoff position that separates observations from queries.
                Only positions before this index contribute to the observation count for each curve.
        Returns:
            tuple: A tuple containing:
                - cutoff_per_curve (np.ndarray): Integer array of shape (seq_len,) indicating the number
                  of observations allocated to each curve. Values are cumulative counts of positions
                  before single_eval_pos.
                - epochs_per_curve (np.ndarray): Integer array of shape (seq_len,) indicating the total
                  number of epochs (observations + queries) allocated to each curve.
                - ordering (np.ndarray): Integer array of shape (seq_len,) representing the sampled
                  ordering of curve indices for each position in the sequence. Will be used to map positions
                  to curves.
        Notes:
            - Uses self.seq_len to determine sequence length
            - Uses self.n_levels to define granularity of fidelity levels
            - Uses self.p to define base probabilities for each position
            - The method samples positions without replacement according to weighted probabilities
        """
        
        # determine # observations/queries per curve
        # TODO: also make this a dirichlet thing

        ids = np.arange(self.seq_len)
        all_levels = np.repeat(ids, self.n_levels)
        all_p = np.repeat(self.p, self.n_levels) / self.n_levels
        ordering = np.random.choice(all_levels, p=all_p, size=self.seq_len, replace=False)

        # calculate the cutoff/samples for each curve
        cutoff_per_curve = np.zeros((self.seq_len,), dtype=int)
        epochs_per_curve = np.zeros((self.seq_len,), dtype=int)
        for i in range(self.seq_len):  # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        return cutoff_per_curve, epochs_per_curve, ordering
    


    def map_(self, curve_configs, curves, num_params, single_eval_pos):

        """Determine x and y values for every curve in the sequence.

        Args:
            curve_configs (np.ndarray): Array of shape (self.seq_len, num_params) containing hyperparameter
                configurations for each curve in the sequence.
            curves (callable): A function that takes x values and a curve index, returning y values.
            single_eval_pos (int, optional): The cutoff position separating observations from queries.
                Defaults to 0.
        Returns:
            tuple: A tuple containing:
                - curve_xs (list): List of length seq_len, where each element is an array of x values
                  for the corresponding curve.
                - curve_ys (list): List of length seq_len, where each element is an array of y values
                  for the corresponding curve.
        """
        cutoff_per_curve, epochs_per_curve, ordering = self.sample_abstract_allocation(single_eval_pos)

        epoch = torch.zeros(self.seq_len)
        id_curve = torch.zeros(self.seq_len)
        curve_val = torch.zeros(self.seq_len)
        config = torch.zeros(self.seq_len, num_params)
        
        curve_xs = []
        curve_ys = []
        for cid in range(self.seq_len):  # loop over every curve
            if epochs_per_curve[cid] > 0:
                # determine x (observations + query)
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    x_[: cutoff_per_curve[cid]] = (
                        np.arange(1, cutoff_per_curve[cid] + 1) / self.n_levels
                    )
                if cutoff_per_curve[cid] < epochs_per_curve[cid]:  # queries (if any)
                    x_[cutoff_per_curve[cid] :] = (
                        np.random.choice(
                            np.arange(cutoff_per_curve[cid] + 1, self.n_levels + 1),
                            size=epochs_per_curve[cid] - cutoff_per_curve[cid],
                            replace=False,
                        )
                        / self.n_levels
                    )
                curve_xs.append(x_)
                # determine y's
                y_ = curves(x_, cid)
                curve_ys.append(y_)
            else:
                curve_xs.append(None)
                curve_ys.append(None)

        # construct the batch data element
        curve_counters = torch.zeros(self.seq_len).type(torch.int64)
        for i in range(self.seq_len):
            cid = ordering[i]
            if i < single_eval_pos or curve_counters[cid] > 0:
                id_curve[i] = cid + 1  # reserve ID 0 for queries
            else:
                id_curve[i] = 0  # queries for unseen curves always have ID 0
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            config[i] = torch.from_numpy(curve_configs[cid])
            curve_val[i] = curve_ys[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        x = torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1)
        y = curve_val

        return x, y

    