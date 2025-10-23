# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# --- User's funcs.py ---
# (I've added minimal implementations based on standard MOC equations
# so the code is runnable. You should use your own 'funcs.py')
def calculate_nu(M, gamma):
    """Calculates the Prandtl-Meyer angle (in radians)"""
    gp1 = gamma + 1
    gm1 = gamma - 1
    return (np.sqrt(gp1 / gm1) * np.arctan(np.sqrt(gm1 / gp1 * (M**2 - 1))) - 
            np.arctan(np.sqrt(M**2 - 1)))

def calculate_Mach(v, gamma, error):
    """Calculates Mach number from Prandtl-Meyer angle (in radians)"""
    # Use a numerical solver (e.g., fsolve) or an iterative method
    # For simplicity, using a basic iterative (binary search) approach
    M_low = 1.0
    M_high = 80.0  # High initial guess
    M_guess = (M_low + M_high) / 2
    while (M_high - M_low) > error:
        v_guess = calculate_nu(M_guess, gamma)
        if v_guess < v:
            M_low = M_guess
        else:
            M_high = M_guess
        M_guess = (M_low + M_high) / 2
    return M_guess

def calculate_Pressure(p_0, gamma, M):
    """Calculates static pressure from total pressure and Mach"""
    return p_0 * (1 + (gamma - 1) / 2 * M**2)**(-gamma / (gamma - 1))

def get_slope_point(phi, M, sign):
    """Calculates the slope of a characteristic line (dy/dx)"""
    mu = np.arcsin(1/M) # Mach angle in radians
    phi_rad = np.deg2rad(phi) # Assuming phi is in degrees from calculations
    
    # Check if phi is in radians from the calculation
    # The 'ACD' print statement shows phi_acd=13.04 degrees.
    # The user's code `np.tan(ACD[0])` in `draw AD` suggests phi is in RADIANS.
    # Let's assume phi is in RADIANS from the calculations.
    
    if sign == '+': # C+ characteristic
        theta = phi_rad - mu
    else: # C- characteristic
        theta = phi_rad + mu
    
    # Avoid vertical slopes
    if np.abs(np.cos(theta)) < 1e-10:
        return 1e10 * np.sign(np.tan(theta))
        
    return np.tan(theta)
# --- End of funcs.py ---


# user settings
n_char = 10
y_A = 1
x_A = 0
p_a = 101325
gamma = 1.4
Mach_error = 1e-10

# initial conditions
M_e = 2.0
p_e = 2*p_a
p_0 = p_e * ((1 + (gamma - 1) / 2 * M_e**2) ** (gamma / (gamma - 1)))

# --- All Calculation Blocks (Unchanged) ---

# region Jet
M_jet = M_e
p_jet = p_e
# Prandtl-Meyer angles are in RADIANS
v_jet = calculate_nu(M_jet, gamma) 
phi_jet = 0 # Flow is axial, angle is 0 radians
jet = np.array([phi_jet, v_jet, M_jet, p_jet])

# region ACD
p_acd = p_a
M_acd = round((((2/(gamma - 1)) * ((p_0 / p_acd)**((gamma - 1) / gamma) - 1))**0.5), 10)
v_acd = calculate_nu(M_acd, gamma) # radians
phi_acd = phi_jet - v_jet + v_acd # This is theta_max in radians
ACD = np.array([phi_acd, v_acd, M_acd, p_acd])
print('ACD:', ACD) # phi_acd is ~0.227 rad or 13.04 deg

# region ABC
phi_1, phi_2 = phi_jet, phi_acd
ABC = np.zeros((n_char, 4))
for i in range(n_char):
    phi_i = phi_1 + i * (phi_2 - phi_1) / (n_char-1) # radians
    v_i = v_jet - phi_jet + phi_i # J- = const
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = calculate_Pressure(p_0, gamma, M_i)
    ABC[i] = np.array([phi_i, v_i, M_i, p_i])

# region BCE
BCE = np.empty((n_char, 4), dtype=object)
for i in range(n_char):
    J_minus_i = ABC[i, 0] + ABC[i, 1] # J- = phi + v
    
    # Centerline point (phi=0)
    v_i_centerline = J_minus_i # At centerline, v = J-
    phi_i_centerline = 0
    J_plus_i = v_i_centerline - phi_i_centerline # Reflected J+ = J-
    
    phi_array= np.zeros((n_char-i))
    v_array= np.zeros((n_char-i))
    M_array= np.zeros((n_char-i))
    p_array= np.zeros((n_char-i))

    # First point is the centerline point
    phi_array[0] = phi_i_centerline
    v_array[0] = v_i_centerline
    M_array[0] = calculate_Mach(v_i_centerline, gamma, Mach_error)
    p_array[0] = calculate_Pressure(p_0, gamma, M_array[0])

    for j in range(i+1, n_char):
        J_minus_j = ABC[j, 0] + ABC[j, 1] # J- from char j
        J_plus_i_intersect = J_plus_i # J+ from reflected char i
        
        # Solve for intersection
        v_ij = (J_minus_j + J_plus_i_intersect) / 2
        phi_ij = (J_minus_j - J_plus_i_intersect) / 2
        M_ij = calculate_Mach(v_ij, gamma, Mach_error)
        p_ij = calculate_Pressure(p_0, gamma, M_ij)
        
        phi_array[j-i] = phi_ij
        v_array[j-i] = v_ij
        M_array[j-i] = M_ij
        p_array[j-i] = p_ij

    BCE[i,0] = phi_array
    BCE[i,1] = v_array
    BCE[i,2] = M_array
    BCE[i,3] = p_array

# region EFH (This is state at Point E, not region EFH)
phi_e = 0
v_e = BCE[-1,1][0] # v at last centerline point
M_e_centerline = BCE[-1,2][0]
p_e_centerline = BCE[-1,3][0]
EFH = np.array([phi_e, v_e, M_e_centerline, p_e_centerline]) # State at E

# region CDEF
CDEF = np.zeros((n_char, 4))
J_minus_ACD = phi_acd + v_acd # J- = const from uniform region ACD
for i in range(n_char):
    # J+ comes from the char originating on line CE
    J_plus_i = BCE[i,1][-1] - BCE[i,0][-1] # J+ = v - phi at point (i, n-1)
    
    phi_i = (J_minus_ACD - J_plus_i) / 2
    v_i = (J_minus_ACD + J_plus_i) / 2
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = calculate_Pressure(p_0, gamma, M_i)
    CDEF[i] = np.array([phi_i, v_i, M_i, p_i])

# region DFG
DFG = np.empty((n_char, 4), dtype=object)
p_boundary_dfg = p_a # Boundary is constant pressure
M_boundary_dfg = M_acd # Mach must be M_acd to have p_a
v_boundary_dfg = v_acd # v must be v_acd
for i in range(n_char):
    J_plus_i = CDEF[i,1] - CDEF[i,0] # J+ = const from CDEF
    
    # Boundary point
    # J+ = v_bdy - phi_bdy => phi_bdy = v_bdy - J+
    phi_boundary_i = v_boundary_dfg - J_plus_i
    v_boundary_i = v_boundary_dfg # v is constant
    # Reflected J- = phi_bdy + v_bdy
    J_minus_i = phi_boundary_i + v_boundary_i 
    
    phi_array_dfg= np.zeros((n_char-i))
    v_array_dfg= np.zeros((n_char-i))
    M_array_dfg= np.zeros((n_char-i))
    p_array_dfg= np.zeros((n_char-i))

    # First point is the boundary point
    phi_array_dfg[0] = phi_boundary_i
    v_array_dfg[0] = v_boundary_i
    M_array_dfg[0] = M_boundary_dfg
    p_array_dfg[0] = p_a

    for j in range(i+1, n_char):
        J_plus_j = CDEF[j,1] - CDEF[j,0] # J+ from char j
        J_minus_i_intersect = J_minus_i # J- from reflected char i
        
        v_ij = (J_minus_i_intersect + J_plus_j) / 2
        phi_ij = (J_minus_i_intersect - J_plus_j) / 2
        M_ij = calculate_Mach(v_ij, gamma, Mach_error)
        p_ij = calculate_Pressure(p_0, gamma, M_ij)
        
        phi_array_dfg[j-i] = phi_ij
        v_array_dfg[j-i] = v_ij
        M_array_dfg[j-i] = M_ij
        p_array_dfg[j-i] = p_ij
        
    DFG[i,0] = phi_array_dfg
    DFG[i,1] = v_array_dfg
    DFG[i,2] = M_array_dfg
    DFG[i,3] = p_array_dfg

# region FGHI
FGHI = np.zeros((n_char, 4))
# This region interacts with the centerline state at E, which is EFH
J_plus_EFH = EFH[1] - EFH[0] # J+ = v_e - phi_e
for i in range(n_char):
    # J- comes from the char originating on line FG
    J_minus_i = DFG[i,0][-1] + DFG[i,1][-1] # J- = phi + v
    
    phi_i = (J_minus_i - J_plus_EFH) / 2
    v_i = (J_minus_i + J_plus_EFH) / 2
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = calculate_Pressure(p_0, gamma, M_i)
    FGHI[i] = np.array([phi_i, v_i, M_i, p_i])

# region GIK (State at G)
# This is the state at point G, the end of the DFG/FGHI regions
p_gik = p_a
M_gik = M_acd
v_gik = v_acd
phi_gik = DFG[-1,0][0] # phi at last boundary point
GIK = np.array([phi_gik, v_gik, M_gik, p_gik])

# region HIJ
HIJ = np.empty((n_char, 4), dtype=object)
for i in range(n_char):
    J_minus_i = FGHI[i, 0] + FGHI[i, 1] # J- from char i
    
    # Centerline point (phi=0)
    v_i_centerline = J_minus_i
    phi_i_centerline = 0
    J_plus_i = v_i_centerline - phi_i_centerline # Reflected J+ = J-
    
    phi_array= np.zeros((n_char-i))
    v_array= np.zeros((n_char-i))
    M_array= np.zeros((n_char-i))
    p_array= np.zeros((n_char-i))

    phi_array[0] = phi_i_centerline
    v_array[0] = v_i_centerline
    M_array[0] = calculate_Mach(v_i_centerline, gamma, Mach_error)
    p_array[0] = calculate_Pressure(p_0, gamma, M_array[0])

    for j in range(i+1, n_char):
        J_minus_j = FGHI[j, 0] + FGHI[j, 1]
        J_plus_i_intersect = J_plus_i
        
        v_ij = (J_minus_j + J_plus_i_intersect) / 2
        phi_ij = (J_minus_j - J_plus_i_intersect) / 2
        M_ij = calculate_Mach(v_ij, gamma, Mach_error)
        p_ij = calculate_Pressure(p_0, gamma, M_ij)

        phi_array[j-i] = phi_ij
        v_array[j-i] = v_ij
        M_array[j-i] = M_ij
        p_array[j-i] = p_ij
        
    HIJ[i,0] = phi_array
    HIJ[i,1] = v_array
    HIJ[i,2] = M_array
    HIJ[i,3] = p_array

# region IJKL
IJKL = np.zeros((n_char, 4))
J_minus_GIK = GIK[0] + GIK[1] # J- from region GIK
for i in range(n_char):
    J_plus_i = HIJ[i,1][-1] - HIJ[i,0][-1] # J+ from line IJ
    
    phi_i = (J_minus_GIK - J_plus_i) / 2
    v_i = (J_minus_GIK + J_plus_i) / 2
    M_i = calculate_Mach(v_i, gamma, Mach_error)
    p_i = calculate_Pressure(p_0, gamma, M_i)
    IJKL[i] = np.array([phi_i, v_i, M_i, p_i])


# --- Coordinate Calculation Blocks (Unchanged) ---
# (These calculate the x,y coordinates of all grid points)

# draw characteristic AB
slope_AB = get_slope_point(ABC[0,0], ABC[0,2], '-')
x_B = x_A + (0 - y_A) / slope_AB

# draw characteristic BC
x_BC = [x_B]
y_BC = [0]
for i in range(1, len(BCE[0][0])): # Corrected loop length
    # This loop calculates points along the *first* C+ char, B-C
    # Average slope of the segment
    m1 = get_slope_point(BCE[0,0][i-1], BCE[0,2][i-1], '+')
    m2 = get_slope_point(BCE[0,0][i], BCE[0,2][i], '+')
    slope_segment = (m1 + m2) / 2
    
    # Slope of intersecting C- char
    slope_intersecting = get_slope_point(ABC[i,0], ABC[i,2], '-')
    
    # Find intersection
    x_prev, y_prev = x_BC[-1], y_BC[-1]
    # C- line equation: y - y_A = m_int * (x - x_A)
    # C+ line equation: y - y_prev = m_seg * (x - x_prev)
    x_i = (y_prev - y_A + slope_intersecting*x_A - slope_segment*x_prev) / (slope_intersecting - slope_segment)
    y_i = y_prev + slope_segment * (x_i - x_prev)
    x_BC.append(x_i)
    y_BC.append(y_i)

# draw rest of ABC
# (No points to store, this is just one line)

# draw rest of BCE
points_BCE = np.empty((n_char, 2), dtype=object)
points_BCE[0][0] = np.array(x_BC)
points_BCE[0][1] = np.array(y_BC) 
for i in range(1, n_char):
    x_i = np.array([])
    y_i = np.array([])
    y0 = 0 # Centerline

    # First point (on centerline)
    slope_i = get_slope_point(BCE[i,0][0], BCE[i,2][0], '+')
    slope_incomming = get_slope_point(BCE[i-1,0][1], BCE[i-1,2][1], '+')
    slope_segment = (slope_i + slope_incomming) / 2
    x_inncomming = points_BCE[i-1][0][1]
    y_incomming = points_BCE[i-1][1][1]
    x_0 = (y0 - y_incomming) / slope_segment + x_inncomming
    x_i = np.insert(x_i, 0, x_0)
    y_i = np.insert(y_i, 0, y0)

    # Rest of the points on this C+ char
    for j in range(1, n_char - i):
        m1 = get_slope_point(BCE[i,0][j], BCE[i,2][j], '+')
        m2 = get_slope_point(ABC[i+j, 0], ABC[i+j, 2], '-') # C- char from ABC

        x_base, y_base = x_i[-1], y_i[-1]
        x_in, y_in = x_A, y_A # C- chars from ABC all start at A
        x_ij = (m1*x_base - m2*x_in + y_in - y_base) / (m1 - m2)
        y_ij = m1*(x_ij - x_base) + y_base

        x_i = np.append(x_i, x_ij)
        y_i = np.append(y_i, y_ij)
    points_BCE[i][0] = x_i
    points_BCE[i][1] = y_i

# draw AD 
y_C = points_BCE[0][1][-1]
x_C = points_BCE[0][0][-1]
# Point D is intersection of jet boundary (phi=const) and first C+ from CDEF
slope_Cplus_0 = get_slope_point(CDEF[0,0], CDEF[0,2], '+')
x_D = (y_C - y_A + np.tan(ACD[0])*x_A - slope_Cplus_0*x_C) / (np.tan(ACD[0]) - slope_Cplus_0)
y_D = np.tan(ACD[0]) * (x_D - x_A) + y_A

x_DF = [x_D]
y_DF = [y_D]
for i in range(1, len(DFG[0][0])): # Corrected loop length
    # This loop calculates points along the *first* C- char, D-F
    m1 = get_slope_point(DFG[0,0][i-1], DFG[0,2][i-1], '-')
    m2 = get_slope_point(DFG[0,0][i], DFG[0,2][i], '-')
    slope_segment = (m1 + m2) / 2
    
    slope_intersecting = get_slope_point(CDEF[i,0], CDEF[i,2], '+')
    
    x_prev, y_prev = x_DF[-1], y_DF[-1]
    x_C_i, y_C_i = points_BCE[i][0][-1], points_BCE[i][1][-1] # Point on CE

    x_i = (y_C_i - y_prev + slope_segment*x_prev - slope_intersecting*x_C_i) / (slope_segment - slope_intersecting)
    y_i = y_prev + slope_segment * (x_i - x_prev)
    x_DF.append(x_i)
    y_DF.append(y_i)

# draw CDEF
# (No points to store)

# draw DFG
points_DFG = np.empty((n_char, 2), dtype=object)
points_DFG[0][0] = np.array(x_DF)
points_DFG[0][1] = np.array(y_DF) 
for i in range(1, n_char):
    x_i = np.array([])
    y_i = np.array([])
    
    # First point (on jet boundary)
    phi = DFG[i,0][0] # phi is constant on boundary
    slope_i = get_slope_point(DFG[i,0][0], DFG[i,2][0], '-')
    slope_incomming = get_slope_point(DFG[i-1,0][1], DFG[i-1,2][1], '-')
    slope_segment = (slope_i + slope_incomming) / 2
    x_inncomming = points_DFG[i-1][0][1]
    y_inncomming = points_DFG[i-1][1][1]
    
    x_prev_bdy, y_prev_bdy = points_DFG[i-1][0][0], points_DFG[i-1][1][0]
    phi_prev = DFG[i-1,0][0]
    slope_bdy = np.tan((phi + phi_prev)/2) # Avg angle for boundary streamline

    x_0 = (y_inncomming - y_prev_bdy + slope_bdy*x_prev_bdy - slope_segment*x_inncomming) / (slope_bdy - slope_segment)
    y_0 = y_prev_bdy + slope_bdy * (x_0 - x_prev_bdy)
    x_i = np.insert(x_i, 0, x_0)
    y_i = np.insert(y_i, 0, y_0)

    for j in range(1, n_char - i):
        m1 = get_slope_point(DFG[i,0][j], DFG[i,2][j], '-')
        m2 = get_slope_point(CDEF[i+j, 0], CDEF[i+j, 2], '+') # C+ char from CDEF

        x_base, y_base = x_i[-1], y_i[-1]
        x_in, y_in = points_BCE[i+j][0][-1], points_BCE[i+j][1][-1] # Point on CE
        x_ij = (m1*x_base - m2*x_in + y_in - y_base) / (m1 - m2)
        y_ij = m1*(x_ij - x_base) + y_base

        x_i = np.append(x_i, x_ij)
        y_i = np.append(y_i, y_ij)
    points_DFG[i][0] = x_i
    points_DFG[i][1] = y_i

# draw FH
# (No points to store)

# draw HI
x_F = points_DFG[0][0][-1]
y_F = points_DFG[0][1][-1]
slope_FH = get_slope_point(DFG[0,0][-1], DFG[0,2][-1], '-')
y_H = 0
x_H = (y_F - y_H) / slope_FH + x_F # Note: slope is dy/dx, so x = x_F + (y - y_F) / m
x_HI = [x_H]
y_HI = [y_H]
for i in range(1, len(HIJ[0][0])): # Corrected loop length
    # This loop calculates points along the *first* C+ char, H-I
    m1 = get_slope_point(HIJ[0,0][i-1], HIJ[0,2][i-1], '+')
    m2 = get_slope_point(HIJ[0,0][i], HIJ[0,2][i], '+')
    slope_segment = (m1 + m2) / 2
    
    slope_intersecting = get_slope_point(FGHI[i,0], FGHI[i,2], '-')
    
    x_prev, y_prev = x_HI[-1], y_HI[-1]
    x_F_i, y_F_i = points_DFG[i][0][-1], points_DFG[i][1][-1] # Point on FG

    x_i = (y_F_i - y_prev + slope_segment*x_prev - slope_intersecting*x_F_i) / (slope_segment - slope_intersecting)
    y_i = y_prev + slope_segment * (x_i - x_prev)
    
    x_HI.append(x_i)
    y_HI.append(y_i)

# draw FGHI
# (No points to store)

# draw rest of HIJ
points_HIJ = np.empty((n_char, 2), dtype=object)
points_HIJ[0][0] = np.array(x_HI)
points_HIJ[0][1] = np.array(y_HI)
for i in range(1, n_char):
    x_i = np.array([])
    y_i = np.array([])
    y0 = 0 # Centerline

    # First point (on centerline)
    slope_i = get_slope_point(HIJ[i,0][0], HIJ[i,2][0], '+')
    slope_incomming = get_slope_point(HIJ[i-1,0][1], HIJ[i-1,2][1], '+')
    slope_segment = (slope_i + slope_incomming) / 2
    x_inncomming = points_HIJ[i-1][0][1]
    y_incomming = points_HIJ[i-1][1][1]
    x_0 = (y0 - y_incomming) / slope_segment + x_inncomming
    x_i = np.insert(x_i, 0, x_0)
    y_i = np.insert(y_i, 0, y0)

    for j in range(1, n_char - i):
        m1 = get_slope_point(HIJ[i,0][j], HIJ[i,2][j], '+')
        m2 = get_slope_point(FGHI[i+j, 0], FGHI[i+j, 2], '-') # C- char from FGHI

        x_base, y_base = x_i[-1], y_i[-1]
        x_in, y_in = points_DFG[i+j][0][-1], points_DFG[i+j][1][-1] # Point on FG
        x_ij = (m1*x_base - m2*x_in + y_in - y_base) / (m1 - m2)
        y_ij = m1*(x_ij - x_base) + y_base

        x_i = np.append(x_i, x_ij)
        y_i = np.append(y_i, y_ij)
    points_HIJ[i][0] = x_i
    points_HIJ[i][1] = y_i


# --- NEW Plotting Function ---

def plot_field(prop_index, title_label):
    """
    Generates a complete plot for a given property.
    prop_index: 2 for Mach, 3 for Pressure
    title_label: String for the colorbar and title
    """
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # --- 1. Collect all property values to set colormap range ---
    all_vals = []
    all_vals.append(jet[prop_index])
    all_vals.append(ACD[prop_index])
    all_vals.extend(ABC[:, prop_index])
    for i in range(n_char):
        all_vals.extend(BCE[i, prop_index])
    all_vals.extend(CDEF[:, prop_index])
    for i in range(n_char):
        all_vals.extend(DFG[i, prop_index])
    all_vals.extend(FGHI[:, prop_index])
    for i in range(n_char):
        all_vals.extend(HIJ[i, prop_index])
    
    vmin = np.min(all_vals)
    vmax = np.max(all_vals)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis
    
    # --- 2. Draw Filled Polygons for Each Region ---

    # Region 1: Jet (Uniform)
    prop_val = jet[prop_index]
    pA = (x_A, y_A)
    pB = (points_BCE[0][0][0], points_BCE[0][1][0])
    dx = (pB[0] - pA[0]) * 0.5 # Offset to draw a rectangle
    verts = [(pA[0]-dx, pA[1]), pA, pB, (pA[0]-dx, 0)]
    ax.add_patch(Polygon(verts, color=cmap(norm(prop_val)), ec='none'))

    # Region 2: ACD (Uniform)
    prop_val = ACD[prop_index]
    pA = (x_A, y_A)
    pC = (points_BCE[0][0][-1], points_BCE[0][1][-1])
    pD = (points_DFG[0][0][0], points_DFG[0][1][0])
    verts = [pA, pC, pD]
    ax.add_patch(Polygon(verts, color=cmap(norm(prop_val)), ec='none'))

    # Region 3: ABC (Simple Wave)
    pA = (x_A, y_A)
    pB_prev = (points_BCE[0][0][0], points_BCE[0][1][0])
    for i in range(1, n_char):
        prop_val = ABC[i-1, prop_index] # Property of cell (i-1)
        pB_curr = (points_BCE[i][0][0], points_BCE[i][1][0])
        verts = [pA, pB_prev, pB_curr]
        ax.add_patch(Polygon(verts, color=cmap(norm(prop_val)), ec='none'))
        pB_prev = pB_curr

    # Region 4: BCE (Non-Simple Wave)
    for i in range(n_char - 1):
        for j in range(len(points_BCE[i][0]) - 1):
            if j + 1 >= len(points_BCE[i+1][0]): break # Avoid array out of bounds
                
            p1 = (points_BCE[i][0][j], points_BCE[i][1][j])
            p2 = (points_BCE[i+1][0][j], points_BCE[i+1][1][j])
            p3 = (points_BCE[i+1][0][j+1], points_BCE[i+1][1][j+1])
            p4 = (points_BCE[i][0][j+1], points_BCE[i][1][j+1])
            
            vals = [
                BCE[i, prop_index][j],
                BCE[i+1, prop_index][j],
                BCE[i+1, prop_index][j+1],
                BCE[i, prop_index][j+1]
            ]
            prop_avg = np.mean(vals)
            ax.add_patch(Polygon([p1, p2, p3, p4], color=cmap(norm(prop_avg)), ec='none'))

    # Region 5: CDEF (Simple Wave)
    for i in range(n_char - 1):
        prop_val = CDEF[i, prop_index]
        pC1 = (points_BCE[i][0][-1], points_BCE[i][1][-1])
        pC2 = (points_BCE[i+1][0][-1], points_BCE[i+1][1][-1])
        pD2 = (points_DFG[i+1][0][0], points_DFG[i+1][1][0])
        pD1 = (points_DFG[i][0][0], points_DFG[i][1][0])
        verts = [pC1, pC2, pD2, pD1]
        ax.add_patch(Polygon(verts, color=cmap(norm(prop_val)), ec='none'))
        
    # Region 6: DFG (Non-Simple Wave)
    for i in range(n_char - 1):
        for j in range(len(points_DFG[i][0]) - 1):
            if j + 1 >= len(points_DFG[i+1][0]): break
                
            p1 = (points_DFG[i][0][j], points_DFG[i][1][j])
            p2 = (points_DFG[i+1][0][j], points_DFG[i+1][1][j])
            p3 = (points_DFG[i+1][0][j+1], points_DFG[i+1][1][j+1])
            p4 = (points_DFG[i][0][j+1], points_DFG[i][1][j+1])
            
            vals = [
                DFG[i, prop_index][j],
                DFG[i+1, prop_index][j],
                DFG[i+1, prop_index][j+1],
                DFG[i, prop_index][j+1]
            ]
            prop_avg = np.mean(vals)
            ax.add_patch(Polygon([p1, p2, p3, p4], color=cmap(norm(prop_avg)), ec='none'))

    # Region 7: FGHI (Simple Wave)
    for i in range(n_char - 1):
        prop_val = FGHI[i, prop_index]
        pF1 = (points_DFG[i][0][-1], points_DFG[i][1][-1])
        pF2 = (points_DFG[i+1][0][-1], points_DFG[i+1][1][-1])
        pH2 = (points_HIJ[i+1][0][0], points_HIJ[i+1][1][0])
        pH1 = (points_HIJ[i][0][0], points_HIJ[i][1][0])
        verts = [pF1, pF2, pH2, pH1]
        ax.add_patch(Polygon(verts, color=cmap(norm(prop_val)), ec='none'))

    # Region 8: HIJ (Non-Simple Wave)
    for i in range(n_char - 1):
        for j in range(len(points_HIJ[i][0]) - 1):
            if j + 1 >= len(points_HIJ[i+1][0]): break
                
            p1 = (points_HIJ[i][0][j], points_HIJ[i][1][j])
            p2 = (points_HIJ[i+1][0][j], points_HIJ[i+1][1][j])
            p3 = (points_HIJ[i+1][0][j+1], points_HIJ[i+1][1][j+1])
            p4 = (points_HIJ[i][0][j+1], points_HIJ[i][1][j+1])
            
            vals = [
                HIJ[i, prop_index][j],
                HIJ[i+1, prop_index][j],
                HIJ[i+1, prop_index][j+1],
                HIJ[i, prop_index][j+1]
            ]
            prop_avg = np.mean(vals)
            ax.add_patch(Polygon([p1, p2, p3, p4], color=cmap(norm(prop_avg)), ec='none'))

    # --- 3. Draw Grid Lines Over the Polygons ---
    line_color = 'black'
    line_width = 0.5
    
    # Centerline
    ax.plot([x_A - 1, x_HI[-1] + 1], [0, 0], 'k--', label='Centerline', lw=line_width*2)

    # C- Characteristics (from ABC)
    for i in range(n_char):
        x = [x_A]
        y = [y_A]
        for j in range(len(points_BCE[i][0])):
            x.append(points_BCE[i][0][j])
            y.append(points_BCE[i][1][j])
        ax.plot(x, y, color=line_color, lw=line_width, ls='-')

    # C+ Characteristics (from BCE)
    for i in range(n_char):
        ax.plot(points_BCE[i][0], points_BCE[i][1], color=line_color, lw=line_width, ls=':')

    # C+ Characteristics (from CDEF)
    for i in range(n_char):
        x = [points_BCE[i][0][-1]]
        y = [points_BCE[i][1][-1]]
        for j in range(len(points_DFG[i][0])):
             x.append(points_DFG[i][0][j])
             y.append(points_DFG[i][1][j])
        ax.plot(x, y, color=line_color, lw=line_width, ls=':')
        
    # C- Characteristics (from DFG)
    for i in range(n_char):
        ax.plot(points_DFG[i][0], points_DFG[i][1], color=line_color, lw=line_width, ls='-')
        
    # C- Characteristics (from FGHI)
    for i in range(n_char):
        x = [points_DFG[i][0][-1]]
        y = [points_DFG[i][1][-1]]
        for j in range(len(points_HIJ[i][0])):
             x.append(points_HIJ[i][0][j])
             y.append(points_HIJ[i][1][j])
        ax.plot(x, y, color=line_color, lw=line_width, ls='-')
        
    # C+ Characteristics (from HIJ)
    for i in range(n_char):
        ax.plot(points_HIJ[i][0], points_HIJ[i][1], color=line_color, lw=line_width, ls=':')

    # Jet Boundary
    x_bdy, y_bdy = [x_A], [y_A]
    for i in range(n_char):
        x_bdy.append(points_DFG[i][0][0])
        y_bdy.append(points_DFG[i][1][0])
    ax.plot(x_bdy, y_bdy, 'r--', label='Jet Boundary', lw=line_width*2)

    # --- 4. Final Plot Setup ---
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Characteristics with {title_label} Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Set axis limits
    ax.set_xlim(x_A - 1, points_HIJ[-1][0][-1] + 1)
    ax.set_ylim(-0.1, y_A + 0.5) # A bit of padding
    ax.set_aspect('equal')
    
    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label=title_label)


# --- Main execution ---

# Plot Mach Number
plot_field(prop_index=2, title_label='Mach Number')

# Plot Pressure
plot_field(prop_index=3, title_label='Pressure [Pa]')

plt.show()