import numpy as np
from math import floor, log10
import sys
sys.path.append('..')
from src.sabra_model import run_model, save_data
from src.utils.params import *

def calculate_pertubation():
    """Calculate a random perturbation with a specific norm.
    
    The norm of the error is defined in the parameter seeked_error_norm

    Returns
    -------
    perturb : ndarray
        The random perturbation
    
    """
    error = np.random.rand(2*n_k_vec).astype(np.float64)
    perturb = np.empty(n_k_vec, dtype=np.complex)
    perturb.real = error[:n_k_vec]
    perturb.imag = error[n_k_vec:]
    lambda_factor = seeked_error_norm/np.linalg.norm(perturb)
    perturb = lambda_factor*perturb

    # Perform small test to be noticed if the perturbation is not as expected
    np.testing.assert_almost_equal(np.linalg.norm(perturb), seeked_error_norm,
        decimal=abs(floor(log10(seeked_error_norm))) + 1)

    return perturb

perturb = calculate_pertubation()
u_old += perturb

run_model()
save_data()