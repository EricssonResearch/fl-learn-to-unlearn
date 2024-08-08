"""Modules for convergence evaluation."""

class NumberOfRounds:
    """Number of rounds as the method of termination."""
    def __init__(self, n_rounds):
        self.n_rounds = n_rounds

    def terminate(self, r):
        """Termination method."""
        if r >= self.n_rounds:
            flag = True
        else:
            flag = False
        return flag
