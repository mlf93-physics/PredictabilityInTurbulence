import sys
sys.path.append('..')
import argparse
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
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    time_group = arg_parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument("--time_to_run", type=float)
    time_group.add_argument("--n_turnovers", type=float)
    args = vars(arg_parser.parse_args())

    if args['n_turnovers'] is not None:
        args['time_to_run'] = ceil(eddy0_turnover_time*args['n_turnovers'])   # [s]
        args['Nt'] = int(args['time_to_run']/dt)
    else:
        args['Nt'] = int(args['time_to_run']/dt)

    data_out = np.zeros((int(args['Nt']*sample_rate), n_k_vec + 1),
        dtype=np.complex128)

    profiler.start()
    print('Running sabra model for {:.2f}s'.format(args["Nt"]*dt))
    run_model(u_old, du_array, data_out, args['Nt'])
    profiler.stop()
    print(profiler.output_text())
    save_data(data_out, folder='data', args=args)
