from pathlib import Path
import sys
sys.path.append('..')
import numpy as np
from math import floor, log10
from src.sabra_model.sabra_model import run_model
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
    
    perturb = np.pad(perturb, pad_width=bd_size, mode='constant')

    return perturb

def save_data(data_out, folder="", prefix=""):
    """Save the data to disc."""

    if len(folder) > 0:
        folder = '/' + folder
    
    # Save data
    temp_time_to_run = "{:e}".format(time_to_run)
    np.savetxt(f"""data{folder}/{prefix}udata_ny{ny}_t{temp_time_to_run}_n_f{n_forcing}_f{forcing.real}_j{int(forcing.imag)}.csv""",
                data_out, delimiter=",",
                header=f"""f={forcing}, n_f={n_forcing}, ny={ny},
                            time={time_to_run}, dt={dt}, epsilon={epsilon},
                            lambda={lambda_const}""")

def import_start_u_profile(folder=None):
    file_names = list(Path('data/' + folder).glob('*.csv'))
    # Find reference file
    ref_file = None
    for ifile, file in enumerate(file_names):
        file_name = file.stem
        if file_name.find('ref') >= 0:
            ref_file = file.name

    file_name = f'./data/{folder}/{ref_file}'
    u_init_profile = np.genfromtxt(file_name,
        dtype=np.complex, delimiter=',', skip_header=3 + burn_in_lines,
        max_rows=1)

    # Skip time datapoint and pad array with zeros
    u_init_profile = np.pad(u_init_profile[1:], pad_width=bd_size, mode='constant')

    return u_init_profile

folder = 'ny1e-08_t5_000000e+00_n_f0_f0_15_j0'
time_to_run = 5   # [s]
Nt = int(time_to_run/dt)
burn_in_time = 1
burn_in_lines = int(burn_in_time/dt*sample_rate)
# Define data out array to store what should be saved.
data_out = np.zeros((int(Nt*sample_rate), n_k_vec + 1), dtype=np.complex128)

# Make reference run
print('Running reference')
run_model(u_old, du_array, data_out, Nt)
save_data(data_out, folder=folder, prefix=f'ref_')

# Make perturbations
u_init_profile = import_start_u_profile(folder=folder)
n_runs = 3
# Reset parameters
time_to_run = time_to_run - burn_in_time   # [s]
Nt = int(time_to_run/dt)
for i in range(n_runs):

    perturb = calculate_pertubation()
    u_old = u_init_profile + perturb

    print(f'Running perturbation {i + 1}')
    run_model(u_old, du_array, data_out, Nt)
    save_data(data_out, folder=folder, prefix=f'perturb{i + 1}_')