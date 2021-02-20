import numpy as np
import matplotlib.pyplot as plt
from .params import *

def dev_plot_eigen_mode_analysis(e_values, J_matrix, e_vectors):
    # Plot eigenvalues
    plt.figure()
    plt.scatter(e_values.real, e_values.imag, color='b', marker='x')
    plt.scatter(e_values.real, e_values.conj().imag, color='r', marker='x')
    plt.grid()

    # Plot J_matrix
    plt.figure()
    plt.pcolormesh(J_matrix.real + J_matrix.imag, cmap='Reds')
    plt.ylim(20, 0)
    plt.clim(0, None)
    plt.colorbar()

    # Plot eigenvectors
    plt.figure()
    plt.pcolormesh(e_vectors.real, cmap='Reds')
    plt.colorbar()
    plt.show()

def dev_plot_perturbation_generation(perturb, perturb_temp):
    # Plot the random and the eigenvector scaled perturbation
    lambda_factor_temp = seeked_error_norm/np.linalg.norm(perturb_temp)
    perturb_temp = lambda_factor_temp*perturb_temp
    
    # Plot random perturbation
    plt.plot(perturb_temp.real, 'b-')
    plt.plot(perturb_temp.imag, 'r-')
    # Plot perturbation scaled along the eigenvector
    plt.plot(perturb.real, 'b--')
    plt.plot(perturb.imag, 'r--')
    plt.show()