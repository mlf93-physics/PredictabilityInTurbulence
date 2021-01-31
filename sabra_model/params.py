import numpy as np

# Initialise model
epsilon = 0.5
lambda_const = 2
ny = 0
dt = 1e-7
time_to_run = 1e-3   # [s]
L = 2*np.pi
Nt = int(time_to_run/dt)
sample_rate = 1/50
n_k_vec = 20
Nx = int(lambda_const**n_k_vec)
u0 = 1
bd_size = 2
forcing = 0
n_forcing = 0

factor2 = - epsilon/lambda_const
factor3 = (1 - epsilon)/lambda_const**2

# Define k vector indices
k_vec_temp = np.array([lambda_const**(n + 1) for n in range(n_k_vec)], dtype=np.int64)

