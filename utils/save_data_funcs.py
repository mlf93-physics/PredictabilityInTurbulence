import numpy as np
from src.utils.params import *

def save_data(data_out, folder="", prefix="", perturb_position=None,
    args=None):
    """Save the data to disc."""

    # if len(folder) > 0:
    #     folder = '/' + folder

    if args is None:
        print('Please supply an argument dictionary to save data.')
        exit()
    
    header = f"f={forcing}, n_f={n_forcing}, ny={ny}, " +\
             f"time={args['time_to_run']}, dt={dt}, epsilon={epsilon}, " +\
             f"lambda={lambda_const}, N_data={int(args['Nt']*sample_rate)}"
    
    if perturb_position is not None:
        header += f', perturb_pos={int(perturb_position)}'

    # Save data
    temp_time_to_run = "{:e}".format(args['time_to_run'])
    np.savetxt(f"""{folder}/{prefix}udata_ny{ny}_t{temp_time_to_run}_n_f{n_forcing}_f{forcing.real}_j{int(forcing.imag)}.csv""",
                data_out, delimiter=",",
                header=header)