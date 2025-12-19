import  numpy as np 

class DimensionPrior:
    def __init__(self, maxnum_dimensions):
        """Determine the number of hyperparameters for the current dataset."""
        assert maxnum_dimensions >= 2
        self.maxnum_dimensions = maxnum_dimensions

    def sample(self):
         # beware upper bound is exclusive!
        return np.random.randint( 1, self.maxnum_dimensions - 1 ) 
    
class FidelityPrior:
    def __init__(self) -> None:
        """Determine the number of fidelity levels for the current dataset."""
        self.n_levels = self.sample()

    def sample(self):
        return int(np.round(10 ** np.random.uniform(0, 3)))