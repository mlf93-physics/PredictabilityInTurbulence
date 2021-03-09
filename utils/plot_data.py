import sys
sys.path.append('..')
from pathlib import Path
import argparse
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import floor, log, ceil, sqrt
from params import *
from import_data_funcs import import_data, import_header
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
    # axes[1].set_title(f'Shell velocity vs. time')
    # plt.suptitle(f'Parameters: f={header_dict["f"]}, $\\nu$={header_dict["ny"]:.2e}, time={header_dict["time"]}')
    plt.legend(legend, loc="center right", bbox_to_anchor=(1.6, 0.5))
    plt.subplots_adjust(left=0.086, right=0.805, wspace=0.3)
    # plt.suptitle(f'Shell velocity vs. time; f={header_dict["f"]}, $\\nu$={header_dict["ny"]:.2e}, time={header_dict["time"]}')

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
        ax.set_title(f'Energy spectrum vs. $n_f$; f={header_dict["f"]}, $\\nu$={header_dict["ny"]:.2e}, time={header_dict["time"]}')

def plot_inviscid_quantities(time, u_store, header_dict, ax=None, omit=None,
        path=None):
    # Plot total energy vs time
    energy_vs_time = np.sum(u_store * np.conj(u_store), axis=1).real
    ax.plot(time.real, energy_vs_time, 'k')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    if omit == 'ny':
        ax.set_title(f'Energy over time vs $\\nu$; f={header_dict["f"]}, $n_f$={header_dict["n_f"]}, time={header_dict["time"]}')
    if omit == 'n_f':
        ax.set_title(f'Energy over time vs $n_f$; f={header_dict["f"]}, $\\nu$={header_dict["ny"]:.2e}, time={header_dict["time"]}')
    
    if path is not None:
        file_names = list(Path(path).glob('*.csv'))
        # Find reference file
        ref_file_index = None
        for ifile, file in enumerate(file_names):
            file_name = file.stem
            if file_name.find('ref') >= 0:
                ref_file_index = ifile
        
        if ref_file_index is None:
            raise ValueError('No reference file found in specified directory')

        for ifile, file_name in enumerate(file_names):
            if ifile == ref_file_index:
                continue
            header_dict = import_header(file_name=file_name)
            # print("header_dict['perturb_pos']", header_dict['perturb_pos'])
            # print(energy_vs_time.shape)
            plt.plot(header_dict['perturb_pos']/sample_rate*dt,
                energy_vs_time[int(header_dict['perturb_pos'])], marker='o')

def plot_inviscid_quantities_per_shell(time, u_store, header_dict, ax=None, omit=None,
        path=None, args=None):
    # Plot total energy vs time
    energy_vs_time = np.cumsum((u_store * np.conj(u_store)).real, axis=1)
    ax.plot(time.real, energy_vs_time, 'k')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    if omit == 'ny':
        ax.set_title(f'Energy over time vs $\\nu$; f={header_dict["f"]}, $n_f$={header_dict["n_f"]}, time={header_dict["time"]}')
    elif omit == 'n_f':
        ax.set_title(f'Energy over time vs $n_f$; f={header_dict["f"]}, $\\nu$={header_dict["ny"]:.2e}, time={header_dict["time"]}')
    else:
        ax.set_title(f'Energy over time; f={header_dict["f"]}'+
            f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
            f', time={header_dict["time"]}s')
    
    if path is not None:
        # file_names = list(Path(path).glob('*.csv'))
        # Find reference file
        # ref_file_index = None
        # for ifile, file in enumerate(file_names):
        #     file_name = file.stem
        #     if file_name.find('ref') >= 0:
        #         ref_file_index = ifile
        
        # if ref_file_index is None:
        #     raise ValueError('No reference file found in specified directory')

        perturb_file_names = list(Path(args['path'], args['perturb_folder']).
            glob('*.csv'))
        
        pert_u_stores, perturb_pos_list, perturb_pos_list_legend, header_dict =\
            import_perturbation_velocities(args)

        for ifile, file_name in enumerate(perturb_file_names):
            # if ifile == ref_file_index:
            #     continue
            header_dict = import_header(file_name=file_name)
            # print("header_dict['perturb_pos']", header_dict['perturb_pos'])
            # print(energy_vs_time.shape)
            # print("header_dict['perturb_pos']", header_dict['perturb_pos'])
            # print("energy_vs_time[int(header_dict['perturb_pos'])]", energy_vs_time[int(header_dict['perturb_pos'])])
            # input()
            point_plot = plt.plot(np.ones(n_k_vec)*header_dict['perturb_pos']/sample_rate*dt,
                energy_vs_time[int(header_dict['perturb_pos'])], 'o')

        
            time_array = np.linspace(0, header_dict['time'],
                int(header_dict['time']*sample_rate/dt),
                dtype=np.float64, endpoint=False)
            
            perturbation_energy_vs_time = np.cumsum(((
                pert_u_stores[ifile] + u_store[int(header_dict['perturb_pos']):
                int(header_dict['perturb_pos']) + int(header_dict['N_data']), :]) *
                np.conj(pert_u_stores[ifile] + u_store[int(header_dict['perturb_pos']):
                int(header_dict['perturb_pos']) + int(header_dict['N_data']), :])).real, axis=1)
            ax.plot(time_array + perturb_pos_list[ifile],
                perturbation_energy_vs_time, color=point_plot[0].get_color())
            
            if ifile + 1 >= args['n_files'] and args['n_files'] > 0:
                break


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

def analyse_eddie_turnovertime(u_store, header_dict, args=None):
    fig, axes = plt.subplots(nrows=2, ncols=1)
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
    axes[0].set_title('Eddy frequencies vs k; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s')

    axes[1].plot(k_vec_temp, mean_eddy_turnover, 'k.', label='Eddy turnover time from $||u||$')
    axes[1].plot(k_vec_temp, k_vec_temp**(-2/3), 'k--', label='$k^{-2/3}$')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].grid()
    axes[1].legend()
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Eddy turnover time')
    axes[1].set_title('Eddy turnover time vs k; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s')



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

def plots_related_to_energy(args=None):
    figs = []
    axes = []

    num_plots = 3

    for i in range(num_plots):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        figs.append(fig)
        axes.append(ax)

    legend_ny = []

    for ifile, file_name in enumerate(file_names):
        data_in, header_dict = import_data(file_name, old_header=False)
        time = data_in[:, 0]
        u_store = data_in[:, 1:]

        # Conserning ny
        # plot_energy_spectrum(u_store, header_dict, ax = axes[0], omit='ny')
        # plot_inviscid_quantities(time, u_store, header_dict, ax = axes[1],
        #     omit='ny', path=args['path'])
        plot_inviscid_quantities_per_shell(time, u_store, header_dict, ax = axes[2],
            path=args['path'], args=args)
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

def plot_eddie_freqs(args=None):
    # file_names = ['../data/udata_ny0_t1.000000e+00_n_f0_f0_j0.csv']

    for ifile, file_name in enumerate(file_names):
        data_in, header_dict = import_data(file_name, skip_lines=args['burn_in_lines'])
        time = data_in[:, 0]
        u_store = data_in[:, 1:]

        analyse_eddie_turnovertime(u_store, header_dict, args=args)
    
def analyse_error_norm_vs_time(u_stores, args=None):

    if len(u_stores.keys()) == 0:
        raise IndexError('Not enough u_store arrays to compare.')

    if args['combinations']:
        combinations = [[j, i] for j in range(len(u_stores.keys()))
            for i in range(j + 1) if j != i]
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(combinations)))

        for enum, indices in enumerate(combinations):
            error_norm_vs_time[:, enum] = np.linalg.norm(u_stores[indices[0]]
                - u_stores[indices[1]], axis=1).real
    else:
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(u_stores.keys())))

        for i in range(len(u_stores.keys())):
            error_norm_vs_time[:, i] = np.linalg.norm(u_stores[i], axis=1).real

    return error_norm_vs_time

def plot_shell_error_vs_time(args=None):

    # Force max on files
    max_files = 3
    if args['n_files'] > max_files:
        args['n_files'] = max_files
    
    u_stores, _, perturb_pos_list_legend, header_dict =\
        import_perturbation_velocities(args)

    time_array = np.linspace(0, header_dict['time'], int(header_dict['time']*sample_rate/dt),
        dtype=np.float64, endpoint=False)
    
    for i in range(len(u_stores)):
        plt.figure()
        plt.plot(time_array, np.abs(u_stores[i]))#, axis=1))
        plt.xlabel('Time [s]')
        plt.ylabel('Error')
        plt.yscale('log')
        plt.ylim(1e-16, 10)
        # plt.xlim(0.035, 0.070)
        plt.legend(k_vec_temp)
        plt.title(f'Error vs time; f={header_dict["f"]}'+
            f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
            f', time={header_dict["time"]} | Folder: {args["perturb_folder"]}')

def import_perturbation_velocities(args=None):
    u_stores = {}

    if args['path'] is None:
        raise ValueError('No path specified')
    
    ref_file_name = list(Path(args['path']).glob('*.csv'))
    # Find reference file
    ref_file_index = None
    for ifile, file in enumerate(ref_file_name):
        file_name = file.stem
        if file_name.find('ref') >= 0:
            ref_file_index = ifile
    
    if ref_file_index is None:
        raise ValueError('No reference file found in specified directory')

    perturb_file_names = list(Path(args['path'], args['perturb_folder']).
        glob('*.csv'))

    perturb_pos_list_legend = []
    perturb_pos_list = []
    for ifile, file_name in enumerate(perturb_file_names):
        data_in, header_dict = import_data(file_name)
        ref_data_in, ref_header_dict = import_data(ref_file_name[ref_file_index],
            old_header=False, skip_lines=int(header_dict['perturb_pos']) + 1,
            max_rows=int(header_dict['N_data']))
        

        u_stores[ifile] = data_in[:, 1:] - ref_data_in[:, 1:]
        perturb_pos_list.append(header_dict["perturb_pos"]/sample_rate*dt)
        perturb_pos_list_legend.append(
            f'Start time: {header_dict["perturb_pos"]/sample_rate*dt:.2f}s')
        
        if args['n_files'] is not None and args['n_files'] >= 0:
            if ifile + 1 >= args['n_files']:
                break

    return u_stores, perturb_pos_list, perturb_pos_list_legend, header_dict

def plot_error_norm_vs_time(args=None):
    
    u_stores, perturb_pos_list, perturb_pos_list_legend, header_dict =\
        import_perturbation_velocities(args)

    error_norm_vs_time = analyse_error_norm_vs_time(u_stores, args=args)
    time_array = np.linspace(0, header_dict['time'], int(header_dict['time']*sample_rate/dt),
        dtype=np.float64, endpoint=False)

    
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].plot(time_array, error_norm_vs_time)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Error')
    axes[0].set_yscale('log')
    axes[0].legend(perturb_pos_list_legend)
    axes[0].set_title(f'Error vs time; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]} | Folder: {args["perturb_folder"]}')
    
    # Plot energy below
    ref_file_name = list(Path(args['path']).glob('*.csv'))
    data_in, ref_header_dict = import_data(ref_file_name[0],
        skip_lines=int(header_dict['perturb_pos']) + 1,
        max_rows=int(header_dict['N_data']))
    
    plot_inviscid_quantities(time_array, data_in[:, 1:], ref_header_dict,
        ax = axes[1])


def plot_2D_eigen_mode_analysis(args=None):
    u_init_profiles, perturb_positions, header_dict =\
        import_start_u_profiles(folder=args['path'], args=args)

    _, e_vector_collection, e_value_collection =\
        find_eigenvector_for_perturbation(
            u_init_profiles, dev_plot_active=False,
            n_profiles=args['n_profiles'],
            local_ny=header_dict['ny'])

    # exit()
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
    plt.figure()
    plt.plot(np.cumsum(e_value_collection.real, axis=0)/kolm_sinai_entropy,
        linestyle='-', marker='.')
    plt.xlabel('Lyaponov index')
    plt.ylabel('$\sum_{i=0}^j \\lambda_j$ / H')
    plt.ylim(-3, 1.5)
    plt.legend(perturb_pos_list)
    plt.title(f'Cummulative eigenvalues; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s')

def plot_3D_eigen_mode_analysis(args=None, right_handed=True):
    u_init_profiles, perturb_positions, header_dict = import_start_u_profiles(folder=args['path'],
        args=args)

    _, e_vector_collection, e_value_collection =\
        find_eigenvector_for_perturbation(
            u_init_profiles, dev_plot_active=False,
            n_profiles=args['n_profiles'],
            local_ny=header_dict['ny'])

    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_vector_collection[i] = e_vector_collection[i][:, sort_id]
    
    e_vector_collection = np.array(e_vector_collection)
    
    # Make data.
    shells = np.arange(0, n_k_vec, 1)
    lyaponov_index = np.arange(0, n_k_vec, 1)
    lyaponov_index, shells = np.meshgrid(lyaponov_index, shells)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(lyaponov_index, shells,
        np.mean(np.abs(e_vector_collection)**2, axis=0),
        cmap='Reds')
    ax.set_xlabel('Lyaponov index, $j$')
    ax.set_ylabel('Shell number, $i$')
    ax.set_zlabel('$<|v_i^j|^2>$')
    ax.view_init(elev=27., azim=-21)
    ax.set_title(f'Eigenvectors vs shell numbers; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s, N_tot={args["n_profiles"]*args["n_runs_per_profile"]}')

    if right_handed:
        ax.set_xlim(n_k_vec, 0)

    fig = plt.figure()
    axes = plt.axes()
    plt.pcolormesh(np.mean(np.abs(e_vector_collection)**2, axis=0), cmap='Reds')
    plt.xlabel('Lyaponov index')
    plt.ylabel('Shell number')
    plt.title(f'Eigenvectors vs shell numbers; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s, N_tot={args["n_profiles"]*args["n_runs_per_profile"]}')

    if right_handed:
        plt.xlim(n_k_vec, 0)
        axes.yaxis.tick_right()
        axes.yaxis.set_label_position("right")
        plt.colorbar(pad=0.1)
    else:
        plt.colorbar()

def plot_eigen_vector_comparison(args=None):
    u_init_profiles, perturb_positions, header_dict =\
        import_start_u_profiles(folder=args['path'], args=args)

    _, e_vector_collection, e_value_collection =\
        find_eigenvector_for_perturbation(
            u_init_profiles, dev_plot_active=False,
            n_profiles=args['n_profiles'],
            local_ny=header_dict['ny'])

    # Sort eigenvectors
    for i in range(len(e_value_collection)):
        sort_id = e_value_collection[i].argsort()[::-1]
        e_vector_collection[i] = e_vector_collection[i][:, sort_id]
    
    e_vector_collection = np.array(e_vector_collection, np.complex128)
    # current_e_vectors = np.mean(e_vector_collection, axis=0)
    # current_e_vectors = e_vector_collection[1]

    dev_plot=True
    
    integral_mean_lyaponov_index = 8#int(np.average(np.arange(current_e_vectors.shape[1])
        # , weights=np.abs(current_e_vectors[0, :])))
    
    print('integral_mean_lyaponov_index', integral_mean_lyaponov_index)

    orthogonality_array = np.zeros(e_vector_collection.shape[0]*
        (integral_mean_lyaponov_index - 1), dtype=np.complex128)

    for j in range(e_vector_collection.shape[0]):
        current_e_vectors = e_vector_collection[j]

        for i in range(1, integral_mean_lyaponov_index):
            # print(f'{integral_mean_lyaponov_index - i} x'+
            #     f' {integral_mean_lyaponov_index + i} : ',
            orthogonality_array[j*(integral_mean_lyaponov_index - 1) + (i - 1)] =\
                np.vdot(current_e_vectors[:,
                integral_mean_lyaponov_index - i],
                current_e_vectors[:, integral_mean_lyaponov_index + i])

            if dev_plot:
                fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
                legend = []

                recent_plot = axes[0].plot(current_e_vectors[:,
                    integral_mean_lyaponov_index - i].real, '-')
                recent_color = recent_plot[0].get_color()
                axes[0].plot(current_e_vectors.imag[:,
                    integral_mean_lyaponov_index - i],
                    linestyle='--', color=recent_color)

                recent_plot = axes[1].plot(current_e_vectors[:, 
                    integral_mean_lyaponov_index + i].real, '-')
                recent_color = recent_plot[0].get_color()
                axes[1].plot(current_e_vectors.imag[:, integral_mean_lyaponov_index + i],
                    linestyle='--', color=recent_color)

                plt.xlabel('Shell number, i')
                axes[0].set_ylabel(f'j = {integral_mean_lyaponov_index - i}')
                axes[1].set_ylabel(f'j = {integral_mean_lyaponov_index + i}')
                axes[0].legend(['Real part', 'Imag part'], loc="center right",
                    bbox_to_anchor=(1.15, 0.5))
                plt.subplots_adjust(right=0.852)
                plt.suptitle(f'Eigenvector comparison; f={header_dict["f"]}'+
                f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
                f', time={header_dict["time"]}s')
    
    if dev_plot:
        fig = plt.figure()
        axes = plt.axes()
        plt.pcolormesh(np.mean(np.abs(e_vector_collection)**2, axis=0), cmap='Reds')
        plt.xlabel('Lyaponov index')
        plt.ylabel('Shell number')
        plt.title(f'Eigenvectors vs shell numbers; f={header_dict["f"]}'+
            f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
            f', time={header_dict["time"]}s, N_tot={args["n_profiles"]*args["n_runs_per_profile"]}')
        plt.xlim(n_k_vec, 0)
        axes.yaxis.tick_right()
        axes.yaxis.set_label_position("right")
        plt.colorbar(pad=0.1)

    # Scatter plot
    plt.figure()
    legend = []

    for i in range(integral_mean_lyaponov_index - 1):
        plt.scatter(orthogonality_array[i:-1:(
            integral_mean_lyaponov_index - 1)].real,
            orthogonality_array[i:-1:(
            integral_mean_lyaponov_index - 1)].imag, marker='.')

        legend.append(f'{integral_mean_lyaponov_index - (i + 1)} x {integral_mean_lyaponov_index + (i + 1)}')
        

    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.legend(legend)
    plt.title(f'Orthogonality of pairs of eigenvectors; f={header_dict["f"]}'+
        f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'+
        f', time={header_dict["time"]}s, N_tot={args["n_profiles"]*args["n_runs_per_profile"]}')

def plot_error_energy_spectrum_vs_time_2D(args=None):
    ref_file_name = list(Path(args['path']).glob('*.csv'))[0]
    perturb_file_names = list(Path(args['path'], args['perturb_folder']).
        glob('*.csv'))

    data_in, header_dict = import_data(perturb_file_names[0])
    ref_data_in, ref_header_dict = import_data(ref_file_name,
        skip_lines=int(header_dict['perturb_pos']) + 1,
        max_rows=int(header_dict['N_data']))

    n_divisions = 10
    error_spectra = np.zeros((n_divisions, n_k_vec), dtype=np.float64)

    for i, data_index in enumerate(range(0,
            int(header_dict['N_data']), int(header_dict['N_data']//n_divisions))):
        error_spectra[i, :] = np.abs(ref_data_in[data_index, 1:] -\
            data_in[data_index, 1:]).real
    
    plt.plot(np.arange(0, n_k_vec), error_spectra.T)
    plt.plot(np.arange(0, n_k_vec), k_vec_temp**(-1/3), 'k--', label='$k^{2/3}$')
    plt.yscale('log')
    legend = [f'{item/sample_rate*dt:.1f}' for item in range(0,
            int(header_dict['N_data']),
            int(header_dict['N_data']//n_divisions))]
    plt.legend(legend)
    plt.xlabel('Shell number')
    plt.ylabel('$u_n - u^{\'}_n$')


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", nargs='+', default=None, type=str)
    arg_parser.add_argument("--path", nargs='?', default=None, type=str)
    arg_parser.add_argument("--plot_type", nargs='+', default=None, type=str)
    arg_parser.add_argument("--seed_mode", default=False, type=bool)
    arg_parser.add_argument("--start_time", nargs='+', type=float)
    subparsers = arg_parser.add_subparsers()
    perturb_parser = subparsers.add_parser("perturb_plot", help=
        'Arguments needed for plotting the perturbation vs time plot.')
    perturb_parser.add_argument("--perturb_folder", nargs='?', default=None, type=str)
    perturb_parser.add_argument("--n_files", default=-1, type=int)
    perturb_parser.add_argument("--combinations", action='store_true')
    # eigen_mode_parser = subparsers.add_parser("eigen_mode_plot", help=
    #     'Arguments needed for plotting 3D eigenmode analysis.')
    arg_parser.add_argument("--burn_in_time",
                                   default=0.0,
                                   type=float)
    arg_parser.add_argument("--n_profiles",
                                   default=1,
                                   type=int)
    arg_parser.add_argument("--n_runs_per_profile",
                                   default=1,
                                   type=int)
    arg_parser.add_argument("--time_to_run",
                                   default=0.1,
                                   type=float)

    args = vars(arg_parser.parse_args())

    # Set seed if wished
    if args['seed_mode']:
        np.random.seed(seed=1)

    if 'burn_in_time' in args:
        args['burn_in_lines'] = int(args['burn_in_time']/dt*sample_rate)
    if 'time_to_run' in args:
        args['Nt'] = int(args['time_to_run']/dt*sample_rate)

    if args['source'] is not None:
        # Prepare file names
        file_names = args['source'] if type(args['source']) is list else [args['source']]

    # Perform plotting
    if "shells_vs_time" in args['plot_type']:
        plot_shells_vs_time([0, 1, 2])
    
    if "2D_eddies" in args['plot_type']:
        plot_eddies()
    
    if "eddie_vel_hist" in args['plot_type']:
        plot_eddy_vel_histograms()

    if "eddie_freqs" in args['plot_type']:
        plot_eddie_freqs(args=args)
    
    if "energy_plots" in args['plot_type']:
        plots_related_to_energy(args=args)

    # plots_related_to_forcing()

    if "error_norm" in args['plot_type']:
        if args['path'] is None:
            print('No path specified to analyse error norms.')
        else:
            plot_error_norm_vs_time(args=args)
    
    if "error_spectrum_vs_time" in args['plot_type']:
        plot_error_energy_spectrum_vs_time_2D(args=args)

    if "shell_error" in args['plot_type']:
        if args['path'] is None:
            print('No path specified to analyse shell error.')
        else:
            plot_shell_error_vs_time(args=args)

    if "eigen_mode_plot_3D" in args['plot_type']:
        plot_3D_eigen_mode_analysis(args=args)
    
    if "eigen_mode_plot_2D" in args['plot_type']:
        plot_2D_eigen_mode_analysis(args=args)

    if "eigen_vector_comp" in args['plot_type']:
        plot_eigen_vector_comparison(args=args)

    plt.show()