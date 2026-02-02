from qiskit_algorithms.optimizers import optimizer
from qiskit_algorithms.optimizers.spsa import SPSA
import qiskit_algorithms
import numpy as np


qiskit_algorithms.utils.algorithm_globals.random_seed = 0

def get_optimizer(
        optimizer : str = 'spsa',
        max_iter : int = 100,
        regularization : float = 1e-8,
        callback = None
):
        """Returns a qiskit_algorithms.optimizers.optimizer instance
            Args:
                - optimizer : str, type of qiskit optimizer to be instantiated (only possible option is 'spsa')
                - max_iter : int, max number of optimization iterations (note: not necessarily the number of
                    circuit evalutations; e.g. SPSA requires 2 circuit evaluations.)
                - regularization : 
                - callback function of the form 
                    callback(
                        self,
                        n_evals,
                        params,
                        fval,
                        step_size,
                        accepted
                        )
        """
        if optimizer == 'spsa':
                optimizer = SPSA(
                    maxiter=max_iter,
                    blocking=True,  # Evaluates the circuit a 3rd time to test the updated params;
                                    # accepts the new params only if the loss is improved by at least allowed_increase
                    regularization=regularization,
                    allowed_increase=0.0,       # Sets the increase in loss allowed by the "blocking" to 0.0
                    callback=callback
                    )
                return optimizer


class SPSAHistory:
    def __init__(self):
        self.params = []
        self.values = []

    def callback(
              self,
              n_evals,
              params,
              fval,
              step_size,
              accepted
              ):
        # Save a copy of the parameters and the energy value
        self.params.append(np.copy(params))
        self.values.append(fval)

from qiskit.primitives import Estimator