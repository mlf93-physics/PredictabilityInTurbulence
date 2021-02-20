import sys
sys.path.append('..')
import numpy as np
from numba import njit, types
from pyinstrument import Profiler
from src.sabra_model.runge_kutta4 import runge_kutta4_vec
from src.utils.params import *
from src.utils.save_data_funcs import save_data

profiler = Profiler()

@njit((types.Array(types.complex128, 1, 'C', readonly=False),
       types.Array(types.complex128, 1, 'C', readonly=False),
       types.Array(types.complex128, 2, 'C', readonly=False),
       types.int64))
def run_model(u_old, du_array, data_out, Nt):
    """Execute the integration of the sabra shell model.
    
    Parameters
    ----------
    u_old : ndarray
        The initial shell velocity profile
    du_array : ndarray
        A helper array used to store the current derivative of the shell
        velocities.
    data_out : ndarray
        An array to store samples of the integrated shell velocities.
    
    """
    sample_number = 0
    # Perform calculations
    for i in range(Nt):
        # Save samples for plotting
        if i % int(1/sample_rate) == 0:
            data_out[sample_number, 0] = dt*i + 0j
            data_out[sample_number, 1:] = u_old[bd_size:-bd_size]
            sample_number += 1
        
        # Update u_old
        u_old = runge_kutta4_vec(y0=u_old, h=dt, du=du_array)


if __name__ == "__main__": 
    profiler.start()
    run_model(u_old, du_array, data_out, Nt)
    profiler.stop()
    print(profiler.output_text())
    save_data(data_out)
