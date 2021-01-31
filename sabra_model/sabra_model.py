import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import floor, log, ceil, sqrt
from runge_kutta4 import runge_kutta4_vec
from params import *

def derivative_evaluator(time=None, u_old=None):
    pre_factor = 1j*k_vec_temp
    du = pre_factor * ( u_old.conj()[bd_size+1:-bd_size+1]*
                        u_old[bd_size+2:] +
                        factor2*u_old.conj()[bd_size-1:-bd_size-1]*
                        u_old[bd_size+1:-bd_size+1] +
                        factor3*u_old[:-bd_size-2]*
                        u_old[bd_size-1:-bd_size-1] ) \
            - ny*k_vec_temp**2*u_old[bd_size:-bd_size]

    # Apply forcing
    du[n_forcing] += forcing
    # Perform padding
    du = np.pad(du, pad_width=bd_size, mode='constant')
    return du

def run_model():
    sample_number = 0

    x_array = np.linspace(-L/2, L/2, Nx + 1, endpoint=False)
    data_out = np.zeros((int(Nt*sample_rate), n_k_vec + 1), dtype=np.complex)
        
    initial_k_vec = k_vec_temp**(-1/3)
    u_old = (u0*initial_k_vec).\
        astype(np.complex)

    # Pad arrays
    u_old = np.pad(u_old, pad_width=bd_size, mode='constant')

    # Perform calculations
    for i in range(Nt):
        u_new = runge_kutta4_vec(y0=u_old, h=dt, dydx=derivative_evaluator)
        # Save samples for plotting
        if i % int(1/sample_rate) == 0:
            data_out[sample_number, 0] = dt*i
            data_out[sample_number, 1:] = u_new[bd_size:-bd_size]
            sample_number += 1

        # Update old array with new array
        u_old = u_new

        if i % int(Nt//10) == 0:
            print(f'Process: {i/Nt*100} %')
    
    # Save data
    temp_time_to_run = "{:e}".format(time_to_run)
    np.savetxt(f"""../data/udata_ny{ny}_t{temp_time_to_run}_n_f{n_forcing}_f{int(forcing.real)}_j{int(forcing.imag)}.csv""", data_out,
                delimiter=",",
                header=f"""f={forcing}, n_f={n_forcing}, ny={ny},
                            time={time_to_run}, dt={dt}, epsilon={epsilon},
                            lambda={lambda_const}""")

 
# Run ny
# for ny in [1e-6, 1e-7, 1e-8]:
#     print(f'Running on ny={ny}')
run_model()

# # Reset ny
# ny = 0
# # Run forcing
# forcing = 10 + 10j
# for n_forcing in [0, 2, 4]:
#     print(f'Running on n_forcing={n_forcing}')
#     run_model()

