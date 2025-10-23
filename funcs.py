# Imports
import numpy as np
import matplotlib.pyplot as plt

def calculate_nu(M, gamma):
    nu = np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))
    return nu

def calculate_Mach(nu, gamma, Mach_error):
    guess = 1.5
    error = 5000
    while error>Mach_error:
        F = calculate_nu(guess, gamma) - nu
        dF = ((guess**2-1)**0.5)/(guess + ((gamma - 1)/2)*guess**3)
        guess  -= F/dF
        error = abs(F)
    return guess

def calculate_Pressure(p_0, gamma, M):
    p = p_0 * ((1 + (gamma - 1) / 2 * M**2) ** (-gamma / (gamma - 1)))
    return p

def get_slope_point(phi, M, type):
    mach_angle = np.arcsin(1/M)
    if type == '+':
        slope= np.tan(phi + mach_angle)
    if type == '-':
        slope= np.tan(phi - mach_angle)
    return slope