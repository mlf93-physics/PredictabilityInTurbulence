import sys
sys.path.append('..')
import argparse
from pathlib import Path
import numpy as np
from math import floor, log10
import matplotlib.pyplot as plt
from src.sabra_model.sabra_model import run_model
from src.utils.params import *
from src.utils.save_data_funcs import save_data
from src.utils.import_data_funcs import import_header
from src.utils.dev_plots import dev_plot_eigen_mode_analysis,\
    dev_plot_perturbation_generation

def find_eigenvector_for_perturbation(u_init_profiles, dev_plot_active=False,
        args=None, header=None, perturb_positions=None):
    """Find the eigenvector corresponding to the minimal of the positive
    eigenvalues of the initial vel. profile.
    
    Use the form of the sabra model to perform the calculation of the Jacobian
    matrix. Perform singular-value-decomposition to get the eigenvalues and
    -vectors. Choose the minimal of the positive eigenvalues with respect to
    the real part of the eigenvalue.

    Parameters
    ----------
    u_init_profiles : ndarray
        The initial velocity profiles

    Returns
    -------
    max_e_vector : ndarray
        The eigenvectors corresponding to the minimal of the positive eigenvalues
    
    """
    print('\nFinding the eigenvalues and eigenvectors at the position of the' +
        ' given velocity profiles\n')

    n_profiles = args['n_profiles']
    e_vector_matrix = np.zeros((n_k_vec, n_profiles), dtype=np.complex)

    # Prepare for returning all eigen vectors and values
    e_vector_collection = []
    e_value_collection = []

    # Perform the conjugation
    u_init_profiles_conj = u_init_profiles.conj()
    # Prepare prefactor vector to multiply on J_matrix
    prefactor_reshaped = np.reshape(pre_factor, (-1, 1))
    # Perform calculation for all u_profiles
    for i in range(n_profiles):
        # Calculate the Jacobian matrix
        J_matrix = np.zeros((n_k_vec, n_k_vec), dtype=np.complex)

        # Add k=2 diagonal
        J_matrix += np.diag(
            u_init_profiles_conj[bd_size+1:-bd_size - 1, i], k=2)
        # Add k=1 diagonal
        J_matrix += factor2*np.diag(np.concatenate(([0],
            u_init_profiles_conj[bd_size:-bd_size - 2, i])), k=1)
        # Add k=-1 diagonal
        J_matrix += factor3*np.diag(np.concatenate(([0],
            u_init_profiles[bd_size:-bd_size - 2, i])), k=-1)
        # Add k=-2 diagonal
        J_matrix += factor3*np.diag(
            u_init_profiles[bd_size+1:-bd_size - 1, i], k=-2)


        # Add contribution from derivatives of the complex conjugates:
        J_matrix += np.diag(np.concatenate((u_init_profiles[bd_size + 2:-bd_size, i], [0])), k=1)
        J_matrix += factor2*np.diag(np.concatenate((u_init_profiles[bd_size + 2:-bd_size, i], [0])), k=-1)

        J_matrix = J_matrix*prefactor_reshaped

        # Add the k=0 diagonal
        temp_ny = args['ny'] if header is None else header['ny']
        # J_matrix -= np.diag(temp_ny * k_vec_temp**2, k=0)

        e_values, e_vectors = np.linalg.eig(J_matrix)
        
        e_vector_collection.append(e_vectors)
        e_value_collection.append(e_values)

        positive_e_values_indices = np.argwhere(e_values.real > 0)
        largest_positive_e_value_index = np.argmax(e_values[positive_e_values_indices].real)

        chosen_e_value_index = positive_e_values_indices[
            largest_positive_e_value_index]

        e_vector_matrix[:, i:i+1] = e_vectors[:, chosen_e_value_index]
    
        if dev_plot_active:
            print('Largest positive eigenvalue', e_values[chosen_e_value_index])
            
            dev_plot_eigen_mode_analysis(e_values, J_matrix, e_vectors,
                header=header, perturb_pos=perturb_positions[i])

    return e_vector_matrix, e_vector_collection, e_value_collection


def calculate_perturbations(perturb_e_vectors, dev_plot_active=False,
        args=None):
    """Calculate a random perturbation with a specific norm for each profile.
    
    The norm of the error is defined in the parameter seeked_error_norm

    Parameters
    ----------
    perturb_e_vectors : ndarray
        The eigenvectors along which to perform the perturbations

    Returns
    -------
    perturbations : ndarray
        The random perturbations
    
    """
    n_profiles = args['n_profiles']
    n_runs_per_profile = args['n_runs_per_profile']
    perturbations = np.zeros((n_k_vec + 2*bd_size, n_profiles*
        n_runs_per_profile), dtype=np.complex)

    # Perform perturbation for all eigenvectors
    for i in range(n_profiles*n_runs_per_profile):
        # Generate random error
        error = np.random.rand(2*n_k_vec).astype(np.float64)*2 - 1
        # Reshape into complex array
        perturb = np.empty(n_k_vec, dtype=np.complex)
        perturb.real = error[:n_k_vec]
        perturb.imag = error[n_k_vec:]
        # Copy array for plotting
        perturb_temp = np.copy(perturb)
        
        # Scale random perturbation with the normalised eigenvector
        perturb = perturb*perturb_e_vectors[:, i // n_runs_per_profile]
        # Find scaling factor in order to have the seeked norm of the error
        lambda_factor = seeked_error_norm/np.linalg.norm(perturb)
        # Scale down the perturbation
        perturb = lambda_factor*perturb

        # Perform small test to be noticed if the perturbation is not as expected
        np.testing.assert_almost_equal(np.linalg.norm(perturb), seeked_error_norm,
            decimal=abs(floor(log10(seeked_error_norm))) + 1)

        perturbations[bd_size:-bd_size, i] = perturb

        if dev_plot_active:
            dev_plot_perturbation_generation(perturb, perturb_temp)

    return perturbations

def import_start_u_profiles(folder=None, args=None):
    n_profiles = args['n_profiles']
    n_runs_per_profile = args['n_runs_per_profile']

    # if args['perturb_pos_mode'] == 'same_positions':
    #     print('\nImporting 1 velocity profiles positioned randomly in '+
    #         'reference datafile')

    # elif args['perturb_pos_mode'] == 'rand_positions':

    file_names = list(Path(folder).glob('*.csv'))
    # Find reference file
    ref_file = None
    for ifile, file in enumerate(file_names):
        file_stem = file.stem
        if file_stem.find('ref') >= 0:
            ref_file = file.name

    # Import header info
    header_dict = import_header(folder=folder, file_name=ref_file,
        old_header=False)

    if args['start_time'] is None:
        print(f'\nImporting {n_profiles} velocity profiles randomly positioned '+
        'in reference datafile\n')
        # Generate random start positions
        division_size = int(header_dict["N_data"] - args['burn_in_lines'] -
            args['Nt']*sample_rate)//n_profiles
        rand_division_start = np.random.randint(low=0, high=division_size,
            size=n_profiles)
        positions = np.array([division_size*i + rand_division_start[i] for i in
            range(n_profiles)])

        burn_in = True
    else:
        print(f'\nImporting {n_profiles} velocity profiles positioned as '+
        'requested in reference datafile\n')
        positions = np.array(args['start_time'])*sample_rate/dt

        burn_in = False

    print('\nPositions of perturbation start: ', (positions +
        burn_in*args['burn_in_lines'])/sample_rate*dt, '(in seconds)')

    # Make path to ref file
    file_name = Path(folder, ref_file)
    
    # Prepare u_init_profiles matrix
    u_init_profiles = np.zeros((n_k_vec + 2*bd_size, n_profiles*
        n_runs_per_profile), dtype=np.complex)
    # Import velocity profiles
    for i, position in enumerate(positions):
        temp_u_init_profile = np.genfromtxt(file_name,
            dtype=np.complex, delimiter=',',
            skip_header=np.int64(1 + position + burn_in*args['burn_in_lines']),
            max_rows=1)
        
        # Skip time datapoint and pad array with zeros
        u_init_profiles[bd_size:-bd_size, i:i + n_runs_per_profile] =\
            np.repeat(np.reshape(temp_u_init_profile[1:],
                (temp_u_init_profile[1:].size, 1)), n_runs_per_profile, axis=1)

    return u_init_profiles, positions + burn_in*args['burn_in_lines'], header_dict

def main(args=None):
    args['Nt'] = int(args['time_to_run']/dt)
    args['burn_in_lines'] = int(args['burn_in_time']/dt*sample_rate)

    # Define data out array to store what should be saved.
    # data_out = np.zeros((int(Nt*sample_rate), n_k_vec + 1), dtype=np.complex128)

    # Make reference run
    # print('Running reference')
    # run_model(u_old, du_array, data_out, Nt)
    # save_data(data_out, folder=folder, prefix=f'ref_')

    # Make perturbations
    u_init_profiles, perturb_positions, _ = import_start_u_profiles(folder=args['path'],
        args=args)


    if args['eigen_perturb']:
        perturb_e_vectors, _, _ = find_eigenvector_for_perturbation(
            u_init_profiles[:, 0:-1:args['n_runs_per_profile']],
            dev_plot_active=False, args=args)
    else:
        perturb_e_vectors = np.ones((n_k_vec, args['n_profiles']),
            dtype=np.complex)

    perturbations = calculate_perturbations(perturb_e_vectors,
        dev_plot_active=False, args=args)

    # exit()


    data_out = np.zeros((int(args['Nt']*sample_rate), n_k_vec + 1), dtype=np.complex128)
    u_store_temp = []
    for i in range(args['n_runs_per_profile']*args['n_profiles']):

        u_old = u_init_profiles[:, i] + perturbations[:, i]

        print(f'Running perturbation {i + 1}/' + 
            f"{args['n_profiles']*args['n_runs_per_profile']} | profile" +
            f" {i // args['n_runs_per_profile']}, profile run" +
            f" {i % args['n_runs_per_profile']}")

        run_model(u_old, du_array, data_out, args['Nt'])
        save_data(data_out, folder=args['path'], prefix=f'perturb{i + 1}_',
            perturb_position=perturb_positions[i // args['n_runs_per_profile']],
            args=args)

if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs='+', type=str)
    arg_parser.add_argument("--path", nargs='?', type=str)
    arg_parser.add_argument("--time_to_run", default=0.1, type=float)
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--ny_n", default=19, type=int)
    arg_parser.add_argument("--n_runs_per_profile", default=1, type=int)
    arg_parser.add_argument("--n_profiles", default=1, type=int)
    arg_parser.add_argument("--start_time", nargs='+', type=float)
    arg_parser.add_argument("--eigen_perturb", default=False, type=bool)
    # arg_parser.add_argument("--perturb_pos_mode", default='same_positions', type=str)
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    args = vars(arg_parser.parse_args())

    args['ny'] = (forcing/(lambda_const**(8/3*args['ny_n'])))**(1/2) #1e-8

    if args['start_time'] is not None:
        if args['n_profiles'] > 1:
            np.testing.assert_equal(len(args['start_time']), args['n_profiles'],
                'The number of start times do not equal the number of' +
                ' requested profiles.')
        else:
            np.testing.assert_equal(len(args['start_time']), 1,
                'Too many start times given')


    # Set seed if wished
    if args['seed_mode']:
        np.random.seed(seed=1)

    main(args=args)