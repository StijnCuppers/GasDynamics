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

def plot_characteristics(array):

    phi = array[:, 0]  # radians
    M = array[:, 2]
    mu = np.arcsin(1 / M)

    # Set scaling and starting point
    x0, y0 = 0, 1.0  # nozzle lip at y = 1
    dx = 5.0  # arbitrary scaling for visual clarity

    # Create figure
    plt.figure(figsize=(10, 5))
    plt.axhline(0, color='k', lw=1, label='Centerline')

    # --- Plot C⁺ lines (downward characteristics)
    for i in range(len(phi)):
        angle = phi[i] + mu[i]
        x1 = x0 + dx * np.cos(angle)
        y1 = y0 - dx * np.sin(angle)
        plt.plot([x0, x1], [y0, y1], 'b--')
        x0, y0 = x1, y1  # move along the boundary

    # --- Reset and plot C⁻ lines (upward reflections)
    x0, y0 = 0, 0  # start at centerline reflection point
    for i in range(len(phi)):
        angle = phi[i] - mu[i]
        x1 = x0 + dx * np.cos(angle)
        y1 = y0 + dx * np.sin(angle)
        plt.plot([x0, x1], [y0, y1], 'r--')

    plt.xlabel("x (arbitrary units)")
    plt.ylabel("y (arbitrary units)")
    plt.title("Characteristic Lines in Jet Expansion (n_char = 4)")
    plt.legend(["Centerline", "C⁺ Characteristics", "C⁻ Characteristics"])
    plt.grid(True)
    plt.show()
