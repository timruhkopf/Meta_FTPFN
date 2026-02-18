import torch
import numpy as np

from ppfn.utils.deprecate import deprecated


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
                  of (visibile/training context)observations allocated to each curve.
                  Values are cumulative counts of positions before single_eval_pos.
                - epochs_per_curve (np.ndarray): Integer array of shape (seq_len,) indicating the total
                  number of epochs (observations + queries) allocated to each curve (including budget tokens
                  after the cutoff -- i.e. the queries).
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
        ordering = np.random.choice(
            all_levels, p=all_p, size=self.seq_len, replace=False
        )

        # calculate the cutoff/samples for each curve
        # cutoff_per_curve = np.zeros((self.seq_len,), dtype=int)
        # epochs_per_curve = np.zeros((self.seq_len,), dtype=int)
        # # FIXME: this can be vectorized using np.bicount!
        # #  for the cutoff_per_sequence, we just need to mask/slice ordering < single_eval_pos
        # for i in range(self.seq_len):  # loop over every pos
        #     cid = ordering[i]
        #     epochs_per_curve[cid] += 1
        #     if i < single_eval_pos:
        #         cutoff_per_curve[cid] += 1

        # 1. Total counts for each curve ID across the whole sequence
        epochs_per_curve = np.bincount(ordering, minlength=self.seq_len)

        # 2. Counts for each curve ID only for the 'observations' (before the cutoff)
        cutoff_per_curve = np.bincount(
            ordering[:single_eval_pos], minlength=self.seq_len
        )

        return cutoff_per_curve, epochs_per_curve, ordering

    @deprecated(
        "This method is the original, unoptimized version of parse_allocation_into_sequence. "
        "It is left here for reference and testing purposes, but should not be used in production due to its inefficiency."
    )
    def parse_allocation_into_sequence_slow(
        self, curve_configs, curves, num_params, single_eval_pos, allocation
    ):
        """Determine x and y values for every curve in the sequence.

        Args:
            curve_configs (np.ndarray): Array of shape (self.seq_len, num_params) containing hyperparameter
                configurations for each token in the sequence.
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
        # FIXME: THIS ENTIRE FUNCTION IS SUPER INEFFICIENT, because we loop over every single token
        cutoff_per_curve, epochs_per_curve, ordering = (
            allocation  # self.sample_abstract_allocation(single_eval_pos)
        )

        # NOTES:
        # **curve_configs**: unit cube sampled hyperparameter configurations for every token in the sequence. Theses are
        # candidate configs, from which we can draw observations and queries (repeatedly if more epochs are allocated
        # or none if cutoff is zero for this curve)
        # **cutoff_per_curve**: a mostly empty tensor, which is in reference to the sequence length curve_configs tensor.
        # It tells us (=0) that we don't use this config at all, or (=k) that this particular config will have k epochs
        # visible in the context (observations)
        # **epochs_per_curve**: similar to cutoff_per_curve, but tells us how many total epochs (observations + queries)
        # are allocated to this curve/config in the sequence; i.e. their difference will tell us how many query tokens
        # will be in the future of this curve/config in the sequence.
        # **curves**: callable that maps (x, curve_id) --> y values, where x are fidelity levels in [0, 1] and curve_id
        # is the size of seq_len; i.e. each of the curve_configs has a corresponding curve mapping x --> y (even if not used)
        # **ordering**: a tensor of shape (seq_len,) that tells us which token id is used at every token position in
        # the sequence. To doing a count over the id from left to right tells us which epoch that token corresponds to.
        # Notice, that len(np.unique(ordering)) is the number of unique configs in the sequence (incl. queries)

        epoch = torch.zeros(self.seq_len)
        id_curve = torch.zeros(self.seq_len)
        curve_val = torch.zeros(self.seq_len)

        # Based on the abstract allocation, which merely tells us how many epochs there are for an abstract unique
        # hyperparameter index (which we can look up in curve_configs), we will now collect the fidelity array
        curve_xs = []
        curve_ys = []
        for cid in range(self.seq_len):  # loop over every token
            if epochs_per_curve[cid] > 0:
                # create empty tensor that will hold the fidelity levels at which we want to evaluate a particular
                # curve (hyperparameter config  )
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    # given the allocated budget create the fidelity level tensor for the observations we want to evaluate
                    x_[: cutoff_per_curve[cid]] = (
                        np.arange(1, cutoff_per_curve[cid] + 1) / self.n_levels
                    )

                if cutoff_per_curve[cid] < epochs_per_curve[cid]:  # queries (if any)
                    # beyond the cutoff of this particular curve, we sample future fidelity levels
                    # that we want to use as query points -- and evaluate with the curves callable
                    x_[cutoff_per_curve[cid] :] = (
                        np.random.choice(
                            np.arange(cutoff_per_curve[cid] + 1, self.n_levels + 1),
                            size=epochs_per_curve[cid] - cutoff_per_curve[cid],
                            replace=False,
                        )
                        / self.n_levels
                    )

                curve_xs.append(x_)

                # determine y's, by evaluating this curve at the fidelity levels x_
                y_ = curves(x_, cid)
                curve_ys.append(y_)

            # an unused curve/config
            else:
                curve_xs.append(None)
                curve_ys.append(None)

        # construct the sequence tensors x and y based on the sampled ordering
        # read the note on ordering above for details!
        config = np.zeros((self.seq_len, num_params))
        curve_counters = torch.zeros(self.seq_len).type(torch.int64)

        for i in range(self.seq_len):
            cid = ordering[i]
            if i < single_eval_pos or curve_counters[cid] > 0:
                id_curve[i] = cid + 1  # reserve ID 0 for queries
            else:
                id_curve[i] = 0  # queries for unseen curves always have ID 0
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            config[i] = curve_configs[cid]
            curve_val[i] = curve_ys[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        x = torch.cat(
            [torch.stack([id_curve, epoch], dim=1), torch.from_numpy(config)], dim=1
        )
        y = curve_val

        return x, y

    def parse_allocation_into_sequence(
        self, curve_configs, curves, num_params, single_eval_pos, allocation
    ):
        cutoff_per_curve, epochs_per_curve, ordering = allocation
        seq_len = self.seq_len

        # 1. Pre-calculate epoch indices (k-th appearance of each ID)
        # We use a dictionary or a small loop to match the original's curve_counters
        epoch_indices = np.zeros(seq_len, dtype=int)
        counters = np.zeros(seq_len, dtype=int)
        for i, cid in enumerate(ordering):
            epoch_indices[i] = counters[cid]
            counters[cid] += 1

        # 2. Replicate the EXACT random sampling order for Fidelities (x)
        # To pass the test, we must loop through cid 0...seq_len just like the original
        curve_xs = [None] * seq_len
        for cid in range(seq_len):
            n_epochs = epochs_per_curve[cid]
            if n_epochs > 0:
                x_ = np.zeros(n_epochs)
                n_cutoff = cutoff_per_curve[cid]
                # Observations (Deterministic)
                if n_cutoff > 0:
                    x_[:n_cutoff] = np.arange(1, n_cutoff + 1) / self.n_levels
                # Queries (Random - MUST be called in this order)
                if n_cutoff < n_epochs:
                    x_[n_cutoff:] = (
                        np.random.choice(
                            np.arange(n_cutoff + 1, self.n_levels + 1),
                            size=n_epochs - n_cutoff,
                            replace=False,
                        )
                        / self.n_levels
                    )
                curve_xs[cid] = x_

        # 3. Vectorized Mapping (The "Speed" Part)
        # Now that we have curve_xs, we map them to the sequence using 'ordering'
        # We can't easily vectorize curve_xs[cid][epoch_indices] without padding,
        # but we can do it in one structured loop which is still faster than the original.

        epochs = np.array(
            [curve_xs[ordering[i]][epoch_indices[i]] for i in range(seq_len)]
        )

        # 4. Vectorized ID Curve Logic
        # Original: id_curve[i] = cid + 1 if (i < single_eval_pos or curve_counters[cid] > 0) else 0
        # Translation: A query gets ID 0 ONLY if it's the very first time we see that curve
        # AND it appears at or after the single_eval_pos.

        first_appearance_idx = np.full(seq_len, -1)
        for i, cid in enumerate(ordering):
            if first_appearance_idx[cid] == -1:
                first_appearance_idx[cid] = i

        id_curve = ordering + 1
        # Mask: Is it the first time we see this CID? AND is that index >= single_eval_pos?
        query_id_mask = (first_appearance_idx[ordering] >= single_eval_pos) & (
            np.arange(seq_len) == first_appearance_idx[ordering]
        )
        id_curve[query_id_mask] = 0

        # 5. Vectorized Configs and Values
        config = curve_configs[ordering]

        # Vectorized Y Evaluation (Assuming curves() handles arrays)
        curve_val = np.zeros(seq_len)
        for cid in np.where(epochs_per_curve > 0)[0]:
            mask = ordering == cid
            curve_val[mask] = curves(epochs[mask], cid)

        # Convert to Torch
        x = torch.cat(
            [
                torch.from_numpy(id_curve).float().unsqueeze(1),  # Column 0: ID
                torch.from_numpy(epochs).float().unsqueeze(1),  # Column 1: Fidelity
                torch.from_numpy(config).float(),  # Columns 2+: Config
            ],
            dim=1,
        )

        y = torch.from_numpy(curve_val).float()

        return x, y
