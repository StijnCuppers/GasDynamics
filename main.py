# STIJN CHECK SIGN OF INVARIANTS


# imports
import numpy as np

from funcs import calculate_nu, calculate_Mach

# user settings
n_char = 10
y_A = 1
x_A = 0
p_a = 101325
gamma = 1.4
Mach_error = 1e-6

# initial conditions
M_e = 2.0
p_e = 2*p_a
p_0 = p_e * ((1 + (gamma - 1) / 2 * M_e**2) ** (gamma / (gamma - 1)))


# region Jet
M_jet = M_e
p_jet = p_e
v_jet = calculate_nu(M_jet, gamma)
phi_jet = 0

# region ACD
M_acd = ((2/(gamma - 1)) * ((p_0 / p_a)**((gamma - 1) / gamma) - 1))**0.5
v_acd = calculate_nu(M_acd, gamma)
phi_acd = phi_jet + v_jet - v_acd
p_acd = p_a
ACD = np.array([phi_acd, v_acd, M_acd, p_acd])

# region ABC
phi_1, phi_2 = phi_jet, phi_acd
ABC = np.zeros((n_char, 4))
for i in range(n_char):
    phi_i = phi_1 + i * (phi_2 - phi_1) / (n_char-1)
    v_i = phi_acd + v_acd - phi_i
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = p_0 * ((1 + (gamma - 1) / 2 * M_i**2) ** (-gamma / (gamma - 1)))
    ABC[i] = np.array([phi_i, v_i, M_i, p_i])

# region BCE
BCE = np.zeros((n_char, 4))
for i in range(n_char):
    invariant = ACD[i, 0] - ACD[i, 1]
