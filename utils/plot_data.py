import sys
sys.path.append('..')
from pathlib import Path
import argparse
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import floor, log, ceil, sqrt
from params import *
from import_data_funcs import import_data
from src.lyaponov.lyaponov_exp_estimator import find_eigenvector_for_perturbation,\
                                    import_start_u_profiles

def plot_shells_vs_time(k_vectors_to_plot=None):

    for ifile, file_name in enumerate(file_names):
        data_in, header_dict = import_data(file_name, old_header=False)
        time = data_in[:, 0]
        u_store = data_in[:, 1:]

    fig = plt.figure(figsize=(8, 4))
    axes = []
    axes.append(plt.subplot(1, 2, 1))
    axes.append(plt.subplot(1, 2, 2))
    # n_k_vec = 10

    legend = []
    if k_vectors_to_plot is None:
        k_vectors_to_plot = range(n_k_vec)
    
    for k in k_vectors_to_plot:
        axes[0].plot(time.real, u_store[:, k].real)
        legend.append(f'k = {k_vec_temp[k]}')
    

    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Velocity', fontsize=12)
    axes[0].set_title('Real part')

    for k in k_vectors_to_plot:
        axes[1].plot(time.real, u_store[:, k].imag)
        legend.append(f'k = {k_vec_temp[k]}')

    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Velocity', fontsize=12)
    axes[1].set_title('Imaginary part')
    axes[1].set_title(f'Shell velocity vs. time')
    plt.suptitle(f'Parameters: f={header_dict["f"]}, $\\nu$={header_dict["ny"]}, time={header_dict["time"]}')
    plt.legend(legend, loc="center right", bbox_to_anchor=(1.6, 0.5))
    plt.subplots_adjust(left=0.086, right=0.805, wspace=0.3)
    plt.suptitle(f'Shell velocity vs. time; f={header_dict["f"]}, $\\nu$={header_dict["ny"]}, time={header_dict["time"]}')

    # handles, labels = axes[1].get_legend_handles_labels()
    # plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5))


def plot_energy_spectrum(u_store, header_dict, ax = None, omit=None):
    # Plot energy
    # for i in range(10):
    #     start = i*1000
    #     plt.plot(k_vec_temp, np.mean((u_store[start:(start + 1000)] * np.conj(u_store[start:(start + 1000)])).real, axis=0))
    ax.plot(k_vec_temp, np.mean((u_store[-u_store.shape[0]//4:] * np.conj(u_store[-u_store.shape[0]//4:])).real, axis=0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('k')
    ax.set_ylabel('Energy')
    if omit == 'ny':
        ax.set_title(f'Energy spectrum vs. $\\nu$; f={header_dict["f"]}, $n_f$={header_dict["n_f"]}, time={header_dict["time"]}')
    if omit == 'n_f':
        ax.set_title(f'Energy spectrum vs. $n_f$; f={header_dict["f"]}, $\\nu$={header_dict["ny"]}, time={header_dict["time"]}')

def plot_inviscid_quantities(time, u_store, header_dict, ax=None, omit=None):
    # Plot total energy vs time
    ax.plot(time.real, np.sum((u_store * np.conj(u_store)).real, axis=1))
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    if omit == 'ny':
        ax.set_title(f'Energy over time vs $\\nu$; f={header_dict["f"]}, $n_f$={header_dict["n_f"]}, time={header_dict["time"]}')
    if omit == 'n_f':
        ax.set_title(f'Energy over time vs $n_f$; f={header_dict["f"]}, $\\nu$={header_dict["ny"]}, time={header_dict["time"]}')
    # plt.ylim([1.68, 1.72])

def plot_eddies():
    n_eddies = 20
    plt.figure()
    axes = []
    legend = []
    dplot = 0.06


    for i in range(n_eddies):
        axes.append(plt.subplot(ceil(sqrt(n_eddies)), floor(sqrt(n_eddies)), i + 1))
        axes[-1].plot(u_store[:, i].real - u_store[0, i].real, u_store[:, i].imag)
        axes[-1].plot(u_store[0, i].real - u_store[0, i].real, u_store[0, i].imag, 'ko',
            label='_nolegend_')
        # legend.append(f'k = {k_vec_temp[-i]}')
    plt.xlabel('Re[u]')
    plt.ylabel('Im[u]')
    # plt.legend(legend, loc='center right', bbox_to_anchor=(1.05, 0.5))
    plt.suptitle(f'u eddies; f={forcing}, ny={ny}, runs={Nt}')

def analyse_eddie_turnovertime(u_store, header_dict, axes):
    # Calculate mean eddy turnover time
    mean_u_norm = np.mean(np.sqrt(u_store*np.conj(u_store)).real, axis=0)
    mean_eddy_turnover = 1/(k_vec_temp*mean_u_norm)
    print('mean_eddy_turnover', mean_eddy_turnover)
    eddy_freq = 1/mean_eddy_turnover
    print('eddy_freq', eddy_freq)

    # plt.figure(10)
    axes[0].plot(k_vec_temp, eddy_freq, 'k.', label='Eddy freq. from $||u||$')
    axes[0].plot(k_vec_temp, k_vec_temp**(2/3), 'k--', label='$k^{2/3}$')
    # print('k_vectors_to_plot', k_vectors_to_plot)
    # for i, k in enumerate(k_vectors_to_plot[::-1]):
    #     axes[0].plot(k_vec_temp[k], eddy_freq[k], '.', label='_nolegend_')
        
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].grid()
    axes[0].legend()
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Eddy frequency')
    axes[0].set_title('Eddy frequencies vs k')

    # fig, axes = plt.subplots(5, 4, sharex=True)
    # axes = axes.reshape((1, u_store.shape[1]))[0]
    
    # for i in range(u_store.shape[1]):
        
    #     spectrum = np.fft.fft(u_store[:, i])
    #     power_spec = np.abs(spectrum)**2
    #     freqs = np.fft.fftfreq(u_store.shape[0], d=1/sample_rate*dt)
    #     idx = np.argsort(freqs)[(freqs.size//2 + 1):]
    #     # plt.plot(freq[:spectrum.shape[-1]//2], power_spec.real)
    #     # plt.plot(freq[:spectrum.shape[-1]//2], power_spec.imag)
    #     axes[i].plot(freqs[idx], power_spec[idx]/1e6)
    #     axes[i].set_xscale('log')
    #     axes[i].grid()
    #     axes[i].set_title(f'k = {k_vec_temp[i]}')

    # # plt.yscale('log')
    # # plt.plot(freq, power_spec)
    # # plt.plot(freq, )
    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Frequency', fontsize=12, labelpad=10)
    # plt.ylabel('Power [in $10^6$]', fontsize=12, labelpad=10)
    # plt.title(f"Power spectrum vs. k; f={header_dict['f']}, " +\
    #     f"$n_f$={header_dict['n_f']}, $\\nu$={header_dict['ny']}, " +\
    #     f"time={header_dict['time']}", pad=25)
    # plt.subplots_adjust(hspace=0.44, left=0.062, right=0.95)

def plot_eddy_vel_histograms():
    n_eddies = 2
    plt.figure()
    axes = []
    legend = []
    dplot = 0.06


    for i in range(n_eddies):
        axes.append(plt.subplot(ceil(sqrt(n_eddies)), floor(sqrt(n_eddies)), i + 1))
        axes[-1].hist2d(u_store[:, -(i + 1)].real, u_store[:, -(i + 1)].imag,
            density=True, cmap='Greys')
        # legend.append(f'k = {k_vec_temp[-i]}')
    plt.xlabel('Re[u]')
    plt.ylabel('Probability')
    # plt.legend(legend, loc='center right', bbox_to_anchor=(1.05, 0.5))
    plt.suptitle(f'u eddy prop dist; f={forcing}, ny={ny}, runs={Nt}')

def plots_related_to_energy():
    figs = []
    axes = []

    num_plots = 2

    for i in range(num_plots):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        figs.append(fig)
        axes.append(ax)

    # file_names = ['../data/udata_ny0_t1.000000e+00_n_f0_f0_j0.csv',
    #               '../data/udata_ny1e-08_t1.000000e+00_n_f0_f0_j0.csv',
    #               '../data/udata_ny1e-07_t1.000000e+00_n_f0_f0_j0.csv',
    #               '../data/udata_ny1e-06_t1.000000e+00_n_f0_f0_j0.csv',
    #               '../data/udata_ny1e-05_t1.000000e+00_n_f0_f0_j0.csv']

    legend_ny = []

    for ifile, file_name in enumerate(file_names):
        data_in, header_dict = import_data(file_name, old_header=False)
        time = data_in[:, 0]
        u_store = data_in[:, 1:]

        # Conserning ny
        plot_energy_spectrum(u_store, header_dict, ax = axes[0], omit='ny')
        plot_inviscid_quantities(time, u_store, header_dict, ax = axes[1], omit='ny')
        legend_ny.append(f'$\\nu$ = {header_dict["ny"]}')

    # Plot Kolmogorov scaling
    axes[0].plot(k_vec_temp, k_vec_temp**(-2/3), 'k--')
    axes[1].legend(legend_ny)
    
    legend_ny.append("$k^{-2/3}$")
    axes[0].legend(legend_ny)
    

    for i in range(num_plots):
        axes[i].grid()

def plots_related_to_forcing():
    figs = []
    axes = []

    num_plots = 2

    for i in range(num_plots):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        figs.append(fig)
        axes.append(ax)

    file_names = ['../data/udata_ny0_t1.000000e+00_n_f0_f0_j0.csv',
                  '../data/udata_ny0_t1.000000e+00_n_f0_f10_j10.csv',
                  '../data/udata_ny0_t1.000000e+00_n_f2_f10_j10.csv',
                  '../data/udata_ny0_t1.000000e+00_n_f4_f10_j10.csv']

    legend_forcing = []

    for ifile, file_name in enumerate(file_names):
        data_in, header_dict = import_data(file_name, old_header=False)
        time = data_in[:, 0]
        u_store = data_in[:, 1:]

        # Conserning forcing
        plot_energy_spectrum(u_store, header_dict, ax = axes[0], omit='n_f')
        plot_inviscid_quantities(time, u_store, header_dict, ax = axes[1], omit='n_f')
        if ifile == 0:
            legend_forcing.append('No forcing')
        else:
            legend_forcing.append(f'$n_f$ = {int(header_dict["n_f"])}')
    
    axes[0].plot(k_vec_temp, k_vec_temp**(-2/3), 'k--', label="Kolmogorov; $k^{-2/3}$")
    axes[0].legend(legend_forcing)
    axes[1].legend(legend_forcing)

def plot_eddie_freqs(axes):
    # file_names = ['../data/udata_ny0_t1.000000e+00_n_f0_f0_j0.csv']

    for ifile, file_name in enumerate(file_names):
        data_in, header_dict = import_data(file_name, old_header=False)
        time = data_in[:, 0]
        u_store = data_in[:, 1:]

        analyse_eddie_turnovertime(u_store, header_dict, axes)
    
def analyse_error_norm_vs_time(u_stores):

    if len(u_stores.keys()) == 0:
        raise IndexError('Not enough u_store arrays to compare.')

    # combinations = [[j, i] for j in range(len(u_stores.keys()))
    #     for i in range(j + 1) if j != i]
    # error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(combinations)))
    error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(u_stores.keys())))

    # for enum, indices in enumerate(combinations):
        # error_norm_vs_time[:, enum] = np.linalg.norm(u_stores[indices[0]]
        #     - u_stores[indices[1]], axis=1).real
    for i in range(len(u_stores.keys())):
        error_norm_vs_time[:, i] = np.linalg.norm(u_stores[i], axis=1).real

    return error_norm_vs_time

def plot_error_norm_vs_time(path=None):
    u_stores = {}

    if path is None:
        raise ValueError('No path specified')
    
    file_names = list(Path(path).glob('*.csv'))
    # Find reference file
    ref_file_index = None
    for ifile, file in enumerate(file_names):
        file_name = file.stem
        if file_name.find('ref') >= 0:
            ref_file_index = ifile
    
    if ref_file_index is None:
        raise ValueError('No reference file found in specified directory')

    perturb_pos_list = []
    for ifile, file_name in enumerate(file_names):
        if ifile == ref_file_index:
            continue
        data_in, header_dict = import_data(file_name, old_header=False)
        ref_data_in, ref_header_dict = import_data(file_names[ref_file_index],
            old_header=False, skip_lines=int(header_dict['perturb_pos']),
            max_rows=int(header_dict['N_data']))

        
        u_stores[ifile] = data_in[:, 1:] - ref_data_in[:, 1:]
        perturb_pos_list.append(f'pos. {header_dict["perturb_pos"]}')


    error_norm_vs_time = analyse_error_norm_vs_time(u_stores)

    plt.plot(error_norm_vs_time)
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend(perturb_pos_list)
    plt.title(f'Error vs time; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]}'+
        f', time={header_dict["time"]}')

def plot_2D_eigen_mode_analysis(args=None):
    u_init_profiles, perturb_positions, header_dict = import_start_u_profiles(folder=args['path'],
        args=args)

    _, e_vector_collection, e_value_collection =\
        find_eigenvector_for_perturbation(
            u_init_profiles, dev_plot_active=True, args=args,
            header=header_dict, perturb_positions=perturb_positions)


    perturb_pos_list = []
    # Sort eigenvalues
    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_value_collection[i] = e_value_collection[i][sort_id]

        # Prepare legend
        perturb_pos_list.append(f'Time: {perturb_positions[i]/sample_rate*dt:.1f}s')

    e_value_collection = np.array(e_value_collection, dtype=np.complex).T

    # Calculate Kolmogorov-Sinai entropy, i.e. sum of positive e values 
    positive_e_values_only = np.copy(e_value_collection)
    positive_e_values_only[positive_e_values_only <= 0] = 0 + 0j
    kolm_sinai_entropy = np.sum(positive_e_values_only.real, axis=0)

    # Plot normalised sum of eigenvalues
    plt.plot(np.cumsum(e_value_collection.real, axis=0)/kolm_sinai_entropy,
        linestyle='-', marker='.')
    plt.xlabel('Lyaponov index')
    plt.ylabel('$\sum_{i=0}^j \\lambda_j$ / H')
    plt.legend(perturb_pos_list)
    plt.title(f'Cummulative eigenvalues; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s')

def plot_3D_eigen_mode_analysis(args=None):
    u_init_profiles, perturb_positions, header_dict = import_start_u_profiles(folder=args['path'],
        args=args)

    _, e_vector_collection, e_value_collection =\
        find_eigenvector_for_perturbation(
            u_init_profiles, dev_plot_active=False, args=args,
            header=header_dict, perturb_positions=perturb_positions)

    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_vector_collection[i] = e_vector_collection[i][:, sort_id]
    
    e_vector_collection = np.array(e_vector_collection)
    
    # Make data.
    shells = np.arange(0, n_k_vec, 1)
    lyaponov_index = np.arange(0, n_k_vec, 1)
    shells, lyaponov_index = np.meshgrid(shells, lyaponov_index)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(lyaponov_index, shells, np.mean(np.abs(e_vector_collection)**2, axis=0))
    # ax.set_xlabel('Shell number')
    # ax.set_ylabel('Lyaponov index')
    # ax.set_zlabel('')

    plt.pcolormesh(np.mean(np.abs(e_vector_collection)**2, axis=0), cmap='Reds')
    plt.xlabel('Lyaponov index')
    plt.ylabel('Shell number')
    plt.title(f'Eigenvectors vs shell numbers; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s, N_tot={args["n_profiles"]*args["n_runs_per_profile"]}')
    plt.colorbar()



if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs='+', default=None, type=str)
    arg_parser.add_argument("--path", nargs='?', default=None, type=str)
    arg_parser.add_argument("--plot_type", nargs='+', default=None, type=str)
    subparsers = arg_parser.add_subparsers()
    eigen_mode_parser = subparsers.add_parser("eigen_mode_plot", help=
        'Arguments needed for plotting 3D eigenmode analysis.')
    eigen_mode_parser.add_argument("--burn_in_time",
                                   default=0.0,
                                   required=True,
                                   type=float)
    eigen_mode_parser.add_argument("--n_profiles",
                                   default=1,
                                   required=True,
                                   type=int)
    eigen_mode_parser.add_argument("--n_runs_per_profile",
                                   default=1,
                                   type=int)
    eigen_mode_parser.add_argument("--time_to_run",
                                   default=0.1,
                                   type=float)



    args = vars(arg_parser.parse_args())

    if 'burn_in_time' in args:
        args['burn_in_lines'] = int(args['burn_in_time']/dt*sample_rate)
    if 'time_to_run' in args:
        args['Nt'] = int(args['time_to_run']/dt*sample_rate)

    if args['source'] is not None:
        # Prepare file names
        file_names = args['source'] if type(args['source']) is list else [args['source']]

    # Perform plotting
    if "shells_vs_time" in args['plot_type']:
        plot_shells_vs_time()
    
    if "2D_eddies" in args['plot_type']:
        plot_eddies()
    
    if "eddie_vel_hist" in args['plot_type']:
        plot_eddy_vel_histograms()

    if "eddie_freqs" in args['plot_type']:
        axes = [plt.axes()]
        plot_eddie_freqs(axes)
    
    if "energy_plots" in args['plot_type']:
        plots_related_to_energy()

    # plots_related_to_forcing()

    if "error_norm" in args['plot_type']:
        if args['path'] is None:
            print('No path specified to analyse error norms.')
        else:
            plot_error_norm_vs_time(path=args['path'])

    if "eigen_mode_plot_3D" in args['plot_type']:
        plot_3D_eigen_mode_analysis(args=args)
    
    if "eigen_mode_plot_2D" in args['plot_type']:
        plot_2D_eigen_mode_analysis(args=args)

    plt.show()