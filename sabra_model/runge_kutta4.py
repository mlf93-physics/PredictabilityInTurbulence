import numpy as np
import matplotlib.pyplot as plt

def runge_kutta4_vec(x0=0, y0=0, h=1, dydx=None):
    # Calculate the k's
    k1 = h*dydx(x0, y0)
    k2 = h*dydx(x0 + 1/2*h, y0 + 1/2*k1)
    k3 = h*dydx(x0 + 1/2*h, y0 + 1/2*k2)
    k4 = h*dydx(x0 + h, y0 + k3)
    
    # Update y
    y_new = y0 + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4

    return y_new

