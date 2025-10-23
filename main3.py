# imports
import numpy as np
import matplotlib.pyplot as plt

from funcs import calculate_nu, calculate_Mach, calculate_Pressure, get_slope_point

# user settings
n_char = 8
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
jet = np.array([phi_jet, v_jet, M_jet, p_jet])

# region ACD
p_acd = p_a
M_acd = round((((2/(gamma - 1)) * ((p_0 / p_acd)**((gamma - 1) / gamma) - 1))**0.5), 4)
v_acd = calculate_nu(M_acd, gamma)
phi_acd = phi_jet - v_jet + v_acd
ACD = np.array([phi_acd, v_acd, M_acd, p_acd])
print('ACD:', ACD)

# region ABC
phi_1, phi_2 = phi_jet, phi_acd
ABC = np.zeros((n_char, 4))
for i in range(n_char):
    phi_i = phi_1 + i * (phi_2 - phi_1) / (n_char-1)
    v_i = v_jet - phi_jet + phi_i
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = calculate_Pressure(p_0, gamma, M_i)
    ABC[i] = np.array([phi_i, v_i, M_i, p_i])


# region BCE
BCE = np.empty((n_char, 4), dtype=object)
for i in range(n_char):
    J_minus_i = ABC[i, 0] + ABC[i, 1]
    v_i_0 = J_minus_i

    phi_array= np.zeros((n_char-(i+1)))
    v_array= np.zeros((n_char-(i+1)))
    M_array= np.zeros((n_char-(i+1)))
    p_array= np.zeros((n_char-(i+1)))
    for j in range(i+1, n_char):
        J_minus_intersect = ABC[j, 0] + ABC[j, 1]
        phi_ij = (J_minus_intersect - v_i_0) / 2
        v_ij = (J_minus_intersect + v_i_0) / 2
        M_ij = calculate_Mach(v_ij, gamma, Mach_error)
        p_ij = calculate_Pressure(p_0, gamma, M_ij)
        phi_array[j-(i+1)] = phi_ij
        v_array[j-(i+1)] = v_ij
        M_array[j-(i+1)] = M_ij
        p_array[j-(i+1)] = p_ij
    phi_array2 = np.insert(phi_array, 0, 0)
    v_array2 = np.insert(v_array, 0, v_i_0)
    M_array2 = np.insert(M_array, 0, calculate_Mach(v_i_0, gamma, Mach_error))
    p_array2 = np.insert(p_array, 0, calculate_Pressure(p_0, gamma, M_array2[0]))
    BCE[i,0] = phi_array2
    BCE[i,1] = v_array2
    BCE[i,2] = M_array2
    BCE[i,3] = p_array2

# region EFH
phi_efh = 0
v_efh = phi_acd + v_acd - phi_efh
M_efh = calculate_Mach(v_efh, gamma, Mach_error)
p_efh = calculate_Pressure(p_0, gamma, M_efh)
EFH = np.array([phi_efh, v_efh, M_efh, p_efh])

# region CDEF
CDEF = np.zeros((n_char, 4))
J_minus_ACD = phi_acd + v_acd
for i in range(n_char):
    J_plus_i = BCE[i,1][-1] - BCE[i,0][-1]
    phi_i = (J_minus_ACD - J_plus_i) / 2
    v_i = (J_minus_ACD + J_plus_i) / 2
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = calculate_Pressure(p_0, gamma, M_i)
    CDEF[i] = np.array([phi_i, v_i, M_i, p_i])

# region DFG
DFG = np.empty((n_char, 4), dtype=object)
p_boundary_dfg = p_a
M_boundary_dfg = round((((2/(gamma - 1)) * ((p_0 / p_boundary_dfg)**((gamma - 1) / gamma) - 1))**0.5), 4)
v_boundary_dfg = calculate_nu(M_boundary_dfg, gamma)
for i in range(n_char):
    J_plus_i = CDEF[i,1] - CDEF[i,0]
    phi_boundary_i = v_boundary_dfg - J_plus_i
    phi_array_dfg= np.zeros((n_char-(i+1)))
    v_array_dfg= np.zeros((n_char-(i+1)))
    M_array_dfg= np.zeros((n_char-(i+1)))
    p_array_dfg= np.zeros((n_char-(i+1)))
    for j in range(i+1, n_char):
        J_plus_intersect = CDEF[j,1] - CDEF[j,0]
        phi_ij = (phi_boundary_i + v_boundary_dfg - J_plus_intersect) / 2
        v_ij = (phi_boundary_i + v_boundary_dfg + J_plus_intersect) / 2
        M_ij = calculate_Mach(v_ij, gamma, Mach_error)
        p_ij = calculate_Pressure(p_0, gamma, M_ij)
        phi_array_dfg[j-(i+1)] = phi_ij
        v_array_dfg[j-(i+1)] = v_ij
        M_array_dfg[j-(i+1)] = M_ij
        p_array_dfg[j-(i+1)] = p_ij
    phi_array2 = np.insert(phi_array_dfg, 0, phi_boundary_i)
    v_array2 = np.insert(v_array_dfg, 0, v_boundary_dfg)
    M_array2 = np.insert(M_array_dfg, 0, M_boundary_dfg)
    p_array2 = np.insert(p_array_dfg, 0, p_a)
    DFG[i,0] = phi_array2
    DFG[i,1] = v_array2
    DFG[i,2] = M_array2
    DFG[i,3] = p_array2

# region FGHI
FGHI = np.zeros((n_char, 4))
J_plus_efh = v_efh - phi_efh
for i in range(n_char):
    J_minus_i = DFG[i,0][-1] + DFG[i,1][-1]
    v_i_efh = (J_plus_efh + J_minus_i) / 2
    phi_i_efh = (J_minus_i - J_plus_efh) / 2
    M_i_efh = calculate_Mach(v_i_efh, gamma, Mach_error)
    p_i_efh = calculate_Pressure(p_0, gamma, M_i_efh)
    FGHI[i] = np.array([phi_i_efh, v_i_efh, M_i_efh, p_i_efh])

# region GIK
p_gik = p_a
M_gik = round((((2/(gamma - 1)) * ((p_0 / p_gik)**((gamma - 1) / gamma) - 1))**0.5), 4)
v_gik = calculate_nu(M_gik, gamma)
phi_gik = v_gik - v_efh + phi_efh
GIK = np.array([phi_gik, v_gik, M_gik, p_gik])

# region HIJ
HIJ = np.empty((n_char, 4), dtype=object)
for i in range(n_char):
    J_minus_i = FGHI[i, 0] + FGHI[i, 1]
    v_i_0 = J_minus_i

    phi_array= np.zeros((n_char-(i+1)))
    v_array= np.zeros((n_char-(i+1)))
    M_array= np.zeros((n_char-(i+1)))
    p_array= np.zeros((n_char-(i+1)))
    for j in range(i+1, n_char):
        J_minus_intersect = FGHI[j, 0] + FGHI[j, 1]
        phi_ij = (J_minus_intersect - v_i_0) / 2
        v_ij = (J_minus_intersect + v_i_0) / 2
        M_ij = calculate_Mach(v_ij, gamma, Mach_error)
        p_ij = calculate_Pressure(p_0, gamma, M_ij)
        phi_array[j-(i+1)] = phi_ij
        v_array[j-(i+1)] = v_ij
        M_array[j-(i+1)] = M_ij
        p_array[j-(i+1)] = p_ij
    phi_array2 = np.insert(phi_array, 0, 0)
    v_array2 = np.insert(v_array, 0, v_i_0)
    M_array2 = np.insert(M_array, 0, calculate_Mach(v_i_0, gamma, Mach_error))
    p_array2 = np.insert(p_array, 0, calculate_Pressure(p_0, gamma, M_array2[0]))
    HIJ[i,0] = phi_array2
    HIJ[i,1] = v_array2
    HIJ[i,2] = M_array2
    HIJ[i,3] = p_array2

# region IJKL
IJKL = np.zeros((n_char, 4))
J_minus_EFH = phi_efh + v_efh
for i in range(n_char):
    J_plus_i = HIJ[i,1][-1] - HIJ[i,0][-1]
    phi_i = (J_minus_EFH - J_plus_i) / 2
    v_i = (J_minus_EFH + J_plus_i) / 2
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = calculate_Pressure(p_0, gamma, M_i)
    IJKL[i] = np.array([phi_i, v_i, M_i, p_i])

# print('jet:', jet)
# print('ACD:', ACD)
#print('ABC:', ABC)
# print('BCE:', BCE)
# print('EFH:', EFH)
# print('CDEF:', CDEF)
# print('DFG:', DFG)
# print('FGHI:', FGHI)
# print('GIK:', GIK)
# print('HIJ:', HIJ)
# print('IJKL:', IJKL)

# draw centerline
y_centerline = 0
x0 = x_A
x1 = x_A + 15
plt.figure(figsize=(12, 5))
plt.plot([x0, x1], [y_centerline, y_centerline], 'k--', label='Centerline')

# draw characteristic AB
slope_AB = get_slope_point(ABC[0,0], ABC[0,2], '-')
x_B = x_A + (0 - y_A) / slope_AB

plt.plot([x_A, x_B], [y_A, 0], 'b-')  # 'b-' = blue solid line
plt.scatter([x_A], [y_A], color='red')
plt.scatter([x_B], [0], color='red')

# draw characteristic BC
# x_BC = [x_B]
# y_BC = [0]
# for i in range(1, len(BCE)):
#     slope_segment = (get_slope_point(BCE[0,0][i-1], BCE[0,2][i-1], '+') + get_slope_point(BCE[0,0][i], BCE[0,2][i], '+')) / 2
#     slope_intersecting = get_slope_point(ABC[i,0], ABC[i,2], '-')
#     x_i = (1-0+slope_segment*x_BC[-1]) / (slope_segment - slope_intersecting)
#     y_i = slope_intersecting * x_i + 1
#     x_BC.append(x_i)
#     y_BC.append(y_i)

# draw characteristic BC
x_BC = [x_B]
y_BC = [0]
for i in range(1, len(BCE)):
    slope_segment = (get_slope_point(BCE[0,0][i-1], BCE[0,2][i-1], '+') + get_slope_point(BCE[0,0][i], BCE[0,2][i], '+')) / 2
    slope_intersecting = get_slope_point(ABC[i,0], ABC[i,2], '-')
    print('slope_intersecting:', slope_intersecting)
    print('slope_segment:', slope_segment)
    # x_i = ((y_BC[-1]-1)/slope_segment - x_BC[-1]) / (slope_intersecting/slope_segment - 1) 
    # y_i = slope_segment * (x_i - x_BC[-1]) + y_BC[-1]
    x_i = (1 - y_BC[-1] + slope_segment * x_BC[-1]) / (slope_segment - slope_intersecting)
    y_i = slope_intersecting * x_i + 1
    x_BC.append(x_i)
    y_BC.append(y_i)
# x_BC = [x_B]
# y_BC = [0]
# for i in range(1, len(BCE)):
#     m_1 = (get_slope_point(BCE[0,0][i], BCE[0,2][i], '+') + get_slope_point(BCE[0,0][i-1], BCE[0,2][i-1], '+')) / 2
    
#     m_2 = (get_slope_point(BCE[0,0][i], BCE[0,2][i], '-') + get_slope_point(ABC[i,0], ABC[i,2], '-')) / 2
    
#     x_base = x_BC[-1]
#     y_base = y_BC[-1]
#     x_incomming = 0  
#     y_incomming = 1  
    
#     y_i = (-y_incomming + m_2*(x_incomming - x_base + y_base))/(m_2/m_1 - 1)
#     x_i = (y_i - y_incomming)/m_2 + x_incomming
    
#     x_BC.append(x_i)
#     y_BC.append(y_i)

plt.plot(x_BC, y_BC, 'g-')
plt.scatter(x_BC[-1], y_BC[-1], color='red')

# draw rest of ABC
for i in range(1, len(ABC)):
    x_i = [0, x_BC[i]]
    y_i = [1, y_BC[i]]
    plt.plot(x_i, y_i, 'b-')

# draw rest of BCE
points_BCE = np.empty((n_char, 2), dtype=object)
points_BCE[0][0] = np.array(x_BC)
points_BCE[0][1] = np.array(y_BC) 
y_i = y_BC
for i in range(1, n_char):
    x_i = np.array([])
    y_i = np.array([])
    y0 = 0
    slope_i = get_slope_point(BCE[i,0][0], BCE[i,2][0], '-')
    slope_incomming = get_slope_point(BCE[i-1,0][1], BCE[i-1,2][1], '-')
    slope_segment = (slope_i + slope_incomming) / 2
    x_inncomming = points_BCE[i-1][0][1]
    y_incomming = points_BCE[i-1][1][1]
    x_0 = (y0 - y_incomming) / slope_segment + x_inncomming
    x_i = np.insert(x_i, 0, x_0)
    y_i = np.insert(y_i, 0, y0)
    for j in range(1, n_char-i):
        # m_1 = (get_slope_point(BCE[i,0][j], BCE[i,2][j], '+') + get_slope_point(BCE[i,0][j-1], BCE[i,2][j-1], '+')) / 2
        # m_2 = (get_slope_point(BCE[i,0][j], BCE[i,2][j], '-') + get_slope_point(BCE[i-1,0][j+1], BCE[i-1,2][j+1], '-')) / 2
        m_1 = get_slope_point(BCE[i,0][j], BCE[i,2][j], '+')
        m_2 = get_slope_point(BCE[i,0][j], BCE[i,2][j], '-')
        y_base = y_i[-1]
        x_base = x_i[-1]
        x_incomming = points_BCE[i-1,0][j+1]
        y_incomming = points_BCE[i-1,1][j+1]
        y_ij = (-y_incomming + m_2*(x_incomming - x_base + y_base))/(m_2/m_1 - 1)
        x_ij = (y_ij - y_incomming)/m_2 + x_incomming
        x_i = np.insert(x_i, j, x_ij)
        y_i = np.insert(y_i, j, y_ij)
    points_BCE[i][0] = x_i
    points_BCE[i][1] = y_i
plt.scatter(points_BCE[-1][0], points_BCE[-1][1], color='red')

for i in range(len(points_BCE)):
    plt.plot(points_BCE[i][0], points_BCE[i][1], 'g-')

n_char = len(points_BCE)

for start_j in range(1, len(points_BCE[0][0])): 
    x = []
    y = []
    for i in range(n_char):
        j = start_j - i
        if j < 0 or j >= len(points_BCE[i][0]):
            break
        x.append(points_BCE[i][0][j])
        y.append(points_BCE[i][1][j])
    plt.plot(x, y, 'g-')

# draw AD and CD
print('points_BCE:', points_BCE)
print('BCE', BCE)
y_E = 3
for i in range(len(CDEF)):
    x_C = points_BCE[i][0][-1]
    y_C = points_BCE[i][1][-1]
    slope = get_slope_point(CDEF[i,0], CDEF[i,2], '+')
    x_i = x_C + (y_E - y_C) / slope
    plt.plot([x_C, x_i], [y_C, y_E], 'r--')
x_E = x_C - (y_C - y_E) / get_slope_point(CDEF[0,0], CDEF[0,2], '+')
x_D = (y_C-y_A+np.tan(ACD[0])*x_A-get_slope_point(CDEF[0,0], CDEF[0,2], '+')*x_C) / (np.tan(ACD[0])-get_slope_point(CDEF[0,0], CDEF[0,2], '+'))
y_D = get_slope_point(CDEF[0,0], CDEF[0,2], '+') * (x_D - x_C) + y_C
plt.plot([x_A, x_D], [y_A, y_D], 'r--', label='jet boundary')  
# Add labels and grid
plt.xlabel('x')
plt.ylabel('y')
plt.title('Characteristics')
plt.legend()
plt.grid(True)

plt.show()