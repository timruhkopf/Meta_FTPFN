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
        cutoff_per_curve = np.bincount(ordering[:single_eval_pos], minlength=self.seq_len)

        return cutoff_per_curve, epochs_per_curve, ordering
    


    def parse_allocation_into_sequence(self, curve_configs, curves, num_params, single_eval_pos, allocation):

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
        # FIXME: THIS ENTIRE FUNCTION IS SUPER INEFFICIENT, because we loop over every single token 
        cutoff_per_curve, epochs_per_curve, ordering = allocation # self.sample_abstract_allocation(single_eval_pos)

        epoch = torch.zeros(self.seq_len)
        id_curve = torch.zeros(self.seq_len)
        curve_val = torch.zeros(self.seq_len)
        config = np.zeros((self.seq_len, num_params))
        
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
            config[i] = curve_configs[cid]
            curve_val[i] = curve_ys[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        x = torch.cat([torch.stack([id_curve, epoch], dim=1), torch.from_numpy(config)], dim=1)
        y = curve_val

        return x, y



    def parse_allocation_into_sequence_vectorized(self, curve_configs, curves, num_params, single_eval_pos, allocation):
        raise NotImplementedError('The below implementation is not tested for quality yet!')
        
        cutoff_per_curve, epochs_per_curve, ordering = allocation
        seq_len = self.seq_len

        # --- LOOP 1: Data Generation (Per Curve) ---
        # We still need to generate the actual points for each curve
        curve_xs = [None] * seq_len
        curve_ys = [None] * seq_len
        
        for cid in range(seq_len):
            n_epochs = epochs_per_curve[cid]
            if n_epochs > 0:
                n_obs = cutoff_per_curve[cid]
                x_ = np.zeros((n_epochs,))
                
                # Observations: linear space
                if n_obs > 0:
                    x_[:n_obs] = np.arange(1, n_obs + 1) / self.n_levels
                
                # Queries: random unique sampling
                if n_obs < n_epochs:
                    remaining_pool = np.arange(n_obs + 1, self.n_levels + 1)
                    x_[n_obs:] = np.random.choice(remaining_pool, size=n_epochs - n_obs, replace=False) / self.n_levels
                
                curve_xs[cid] = x_
                curve_ys[cid] = curves(x_, cid)

        # --- VECTORIZED ASSEMBLY (Replacing Loop 3) ---
        
        # 1. Calculate the "Instance Rank" (Cumulative Count)
        # This tells us: "Is this the 1st, 2nd, or N-th time we've seen this cid?"
        # We use a sorting trick to get cumulative counts in O(N log N)
        sort_idx = np.argsort(ordering)
        sorted_ordering = ordering[sort_idx]
        # Find where the CID changes in the sorted array
        changes = np.concatenate(([0], np.where(sorted_ordering[:-1] != sorted_ordering[1:])[0] + 1))
        # Create the counts and map them back to original order
        counts = np.arange(len(ordering)) - np.repeat(changes, np.diff(np.concatenate((changes, [len(ordering)]))))
        instance_rank = np.zeros_like(ordering)
        instance_rank[sort_idx] = counts

        # 2. Vectorized ID Assignment (ID 0 logic)
        # Mask: Curve is 0 if it's the FIRST time seeing it AND it's in the query section
        _, first_occurrence_indices = np.unique(ordering, return_index=True)
        is_first_occurrence = np.zeros(seq_len, dtype=bool)
        is_first_occurrence[first_occurrence_indices] = True
        
        is_query_pos = np.arange(seq_len) >= single_eval_pos
        
        id_curve = ordering + 1  # Standard IDs
        id_curve[is_first_occurrence & is_query_pos] = 0 # Apply anonymity mask

        # 3. Extract values using the instance_rank
        # Since curve_xs is a list of arrays, we use the pre-calculated rank to index
        epoch = np.array([curve_xs[cid][rank] for cid, rank in zip(ordering, instance_rank)])
        curve_val = np.array([curve_ys[cid][rank] for cid, rank in zip(ordering, instance_rank)])
        
        # 4. Construct Tensors
        id_curve = torch.from_numpy(id_curve).float()
        epoch = torch.from_numpy(epoch).float()
        curve_val = torch.from_numpy(curve_val).float()
        config = torch.from_numpy(curve_configs[ordering]).float()

        x = torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1)
        y = curve_val

        return x, y
    

if __name__ == "__main__":
              

    import time
    from ppfn.dataset.prior.multifidelity_problem_prior import MultiFidelityTask 

    def benchmark_parse_allocation():
        # Setup parameters
        seq_len = 2048
        num_params = 10
        n_levels = 100
        single_eval_pos = 1024

        dataset_prior = MultiFidelityTask(num_params, 23)
        dataset_prior.sample_task()
        curve_configs = np.random.uniform(size=(seq_len, num_params)) 

        curves = dataset_prior.get_marginal_curve(torch.from_numpy(curve_configs).float())  # get callable to evaluate (hp, t) --> y

        # --- Execution ---

        prior = AllocationPrior(seq_len, n_levels)
        configs = np.random.randn(seq_len, num_params)
        allocation = prior.sample_abstract_allocation(single_eval_pos)

        # Warmup
        prior.parse_allocation_into_sequence(configs, curves, num_params, single_eval_pos, allocation)
        prior.parse_allocation_into_sequence_vectorized(configs, curves, num_params, single_eval_pos, allocation)

        # Timing Original
        t0 = time.perf_counter()
        x1, y1 = prior.parse_allocation_into_sequence(configs, curves, num_params, single_eval_pos, allocation)
        t1 = time.perf_counter()

        # Timing Vectorized
        t2 = time.perf_counter()
        x2, y2 = prior.parse_allocation_into_sequence_vectorized(configs, curves, num_params, single_eval_pos, allocation)
        t3 = time.perf_counter()

        # Assert equivalence
        # FIXME: check equivalence (due to repeated rnd sampling in loop, this is difficult)
        # assert torch.allclose(x1, x2), "X tensors do not match!"
        # assert torch.allclose(y1, y2), "Y tensors do not match!"

        print(f"Results (seq_len={seq_len}):")
        print(f"Original Time:   {(t1 - t0)*1000:.2f} ms")
        print(f"Vectorized Time: {(t3 - t2)*1000:.2f} ms")
        print(f"Speedup:         {(t1 - t0)/(t3 - t2):.2f}x")
    
    benchmark_parse_allocation()
    # Loading BNN prior ECDF cache from bnn_prior_ecdf.npy
    # Results (seq_len=2048):
    # Original Time:   124.18 ms
    # Vectorized Time: 72.07 ms
    # Speedup:         1.72x

  
    def benchmark_bincount():
        # Setup parameters
        seq_len = 5000
        n_levels = 10
        single_eval_pos = 2000
        p = np.random.dirichlet(np.ones(seq_len), size=1).flatten()  # Random weights summing to 1
        
        # Generate common ordering to ensure we compare apples to apples
        ids = np.arange(seq_len)
        all_levels = np.repeat(ids, n_levels)
        all_p = np.repeat(p, n_levels) / n_levels
        ordering = np.random.choice(all_levels, p=all_p, size=seq_len, replace=False)

        # --- METHOD 1: ORIGINAL LOOP ---
        start_loop = time.perf_counter()
        
        cutoff_loop = np.zeros((seq_len,), dtype=int)
        epochs_loop = np.zeros((seq_len,), dtype=int)
        for i in range(seq_len):
            cid = ordering[i]
            epochs_loop[cid] += 1
            if i < single_eval_pos:
                cutoff_loop[cid] += 1
                
        end_loop = time.perf_counter()

        # --- METHOD 2: VECTORIZED BINCOUNT ---
        start_vec = time.perf_counter()
        
        epochs_vec = np.bincount(ordering, minlength=seq_len)
        cutoff_vec = np.bincount(ordering[:single_eval_pos], minlength=seq_len)
        
        end_vec = time.perf_counter()

        # --- VALIDATION ---
        epochs_match = np.array_equal(epochs_loop, epochs_vec)
        cutoff_match = np.array_equal(cutoff_loop, cutoff_vec)
        
        print(f"Validation:")
        print(f" - Epochs match: {epochs_match}")
        print(f" - Cutoffs match: {cutoff_match}")
        print("-" * 30)
        print(f"Timing (seq_len={seq_len}):")
        print(f" - Original Loop:     {(end_loop - start_loop) * 1000:.4f} ms")
        print(f" - Vectorized Bincount: {(end_vec - start_vec) * 1000:.4f} ms")
        print(f" - Speedup:            {(end_loop - start_loop) / (end_vec - start_vec):.1f}x")
        assert np.array_equal(epochs_loop, epochs_vec) and np.array_equal(cutoff_loop, cutoff_vec)


    # ------------------------------
    # Timing (seq_len=5000):
    # - Original Loop:     2.1054 ms
    # - Vectorized Bincount: 0.0554 ms
    # - Speedup:            38.0x
    # benchmark_bincount()