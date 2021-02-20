import numpy as np
from math import ceil

#### Initialise sabra model constants ####
epsilon = 0.5
lambda_const = 2
dt = 1e-7
sample_rate = 1/1000
eddy0_turnover_time = 2**(-2/3)
time_to_run = ceil(eddy0_turnover_time*50)   # [s]
Nt = int(time_to_run/dt)
burn_in_time = 0.5
burn_in_lines = int(burn_in_time/dt*sample_rate)
n_k_vec = 20
u0 = 1
bd_size = 2
forcing = 1     #0.15
n_forcing = 0
ny_n = 19
ny = (forcing/(lambda_const**(8/3*ny_n)))**(1/2) #1e-8

# Define factors to be used in the derivative calculation
factor2 = - epsilon/lambda_const
factor3 = (1 - epsilon)/lambda_const**2

#### Initialise sabra model arrays ####
# Define k vector indices
k_vec_temp = np.array([lambda_const**(n + 1) for n in range(n_k_vec)], dtype=np.int64)
pre_factor = 1j*k_vec_temp
# Define du array to store derivative
du_array = np.zeros(n_k_vec + 2*bd_size, dtype=np.complex128)
# # Define data out array to store what should be saved.
# data_out = np.zeros((int(Nt*sample_rate), n_k_vec + 1), dtype=np.complex128)

# Calculate initial k and u profile. Put in zeros at the boundaries
initial_k_vec = k_vec_temp**(-1/3)
u_old = (u0*initial_k_vec).astype(np.complex128)
u_old = np.pad(u_old, pad_width=bd_size, mode='constant')

data_out = np.zeros((int(Nt*sample_rate), n_k_vec + 1), dtype=np.complex128)

#### Initialise Lyaponov exponent estimator constants ####
seeked_error_norm = 1e-14
