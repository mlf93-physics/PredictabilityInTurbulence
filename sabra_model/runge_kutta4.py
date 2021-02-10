from numba import njit, types
from utils.params import *

@njit((types.Array(types.complex128, 1, 'C', readonly=True),
       types.Array(types.complex128, 1, 'C', readonly=False)))
def derivative_evaluator(u_old=None, du=None):
    """Derivative evaluator used in the Runge-Kutta method.

    Calculates the derivative of the shell velocities.

    Parameters
    ----------
    u_old : ndarray
        The previous shell velocity array
    du : ndarray
        A helper array used to store the current derivative of the shell
        velocities. Updated at each call to this function.

    Returns
    -------
    du : ndarray
        The updated derivative of the shell velocities
    
    """
    # Calculate change in u (du)
    du[bd_size:-bd_size] = pre_factor * ( u_old.conj()[bd_size+1:-bd_size+1]*
                        u_old[bd_size+2:] +
                        factor2*u_old.conj()[bd_size-1:-bd_size-1]*
                        u_old[bd_size+1:-bd_size+1] +
                        factor3*u_old[:-bd_size-2]*
                        u_old[bd_size-1:-bd_size-1] )\
                        - ny*k_vec_temp**2*u_old[bd_size:-bd_size]

    # Apply forcing
    du[n_forcing + bd_size] += forcing
    return du

@njit(types.Array(types.complex128, 1, 'C', readonly=False)
      (types.Array(types.complex128, 1, 'C', readonly=False),
       types.float64,
       types.Array(types.complex128, 1, 'C', readonly=False)))
def runge_kutta4_vec(y0=0, h=1, du=None):
    """Performs the Runge-Kutta-4 integration of the shell velocities.
    
    Parameters
    ----------
    x0 : ndarray
        The x array
    y0 : ndarray
        The y array
    h : float
        The x step to integrate over.
    du : ndarray
        A helper array used to store the current derivative of the shell
        velocities.

    Returns
    -------
    y0 : ndarray
        The y array at x + h.
    
    """
    # Calculate the k's
    k1 = h*derivative_evaluator(u_old=y0, du=du)
    k2 = h*derivative_evaluator(u_old=y0 + 1/2*k1, du=du)
    k3 = h*derivative_evaluator(u_old=y0 + 1/2*k2, du=du)
    k4 = h*derivative_evaluator(u_old=y0 + k3, du=du)
    
    # Update y
    y0 = y0 + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4

    return y0

