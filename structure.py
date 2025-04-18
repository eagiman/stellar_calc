import numpy as np

from opacity import get_opacity
from constants import * # in cgs units

# Chosen Metallicity
X = 0.7
Y = 0.28
Z = 0.02

def get_mu(X, Y):
    """Get mean molecular weight assuming fully ionized gas

    Parameters
    ----------
    X : float
        Hydrogen fraction
    Y : float
        Helium fraction

    Returns
    -------
    float
        mean molecular weight
    """
    Z = 1 - X - Y

    return 1/(2*X + 3*Y/4 + Z/2)

# Get density rho from pressure, temperature, and composition
def get_density(P, T, X, Y):
    """Get density given pressure and temperature (and metallicity)

    Parameters
    ----------
    P : float
        Pressure in g cm^-1 s^-2
    T : float
        Temperature in K
    X : float
        Hydrogen fraction
    Y : float
        Helium fraction

    Returns
    -------
    float
        density in g cm^-3
    """
    mu = get_mu(X, Y)
    P_rad = (a * T**4)/3

    return (P - P_rad) * (mu * mp)/(k * T)

def get_f11(T, rho):

    T7 = T/1e7
    X_1, X_2, zeta = 1, 1, 1
    EkT = 5.92e-3 * X_1 * X_2 * (rho/T7**3)**(1/2)

    return np.exp(EkT)

def get_pp_energy(T, rho, X, Y):
    """Calculate energy generated by the proton-proton chain

    Parameters
    ----------
    T : float
        Temperature in K
    rho : float
        Density in g cm^-3
    X : float
        Hydrogen fraction
    Y : float
        Helium fraction

    Returns
    -------
    float
        Energy generated by pp chain in g cm^2 s^-2 (erg)
    """
    Z = 1 - X - Y
    T9 = T/1e9
    g11 = 1 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4
    psi = 1.2 # psi based on fig 18.7
    f11 = get_f11(T, rho)

    return 2.57e4 * psi * f11 * g11 * rho * X**2 * T9**(-2/3) * np.exp(-3.381/T9**(1/3))

def get_cno_energy(T, rho, X, Y):
    """Calculate energy generated by the CNO cycle

    Parameters
    ----------
    T : float
        Temperature in K
    rho : float
        Density in g cm^-3
    X : float
        Hydrogen fraction
    Y : float
        Helium fraction

    Returns
    -------
    float
        Energy generated by CNO cycle in g cm^2 s^-2 (erg)
    """
    Z = 1 - X - Y
    T9 = T/1e9
    g141 = 1 - 2*T9 + 3.41*T9**2 - 2.43*T9**3

    return 8.24e25 * g141 * Z * X * rho * T9**(-2/3) * np.exp(-15.231*T9**(-1/3) - (T9/0.8)**2)

def get_energy(T, rho, X, Y):
    """Calculate total energy generated by nuclear reactions

    Parameters
    ----------
    T : float
        Temperature in K
    rho : float
        Density in g cm^-3
    X : float
        Hydrogen fraction
    Y : float
        Helium fraction

    Returns
    -------
    float
        Sum of energy produced by pp chain and cno cycle in g cm^2 s^-2 (erg)
    """
    pp = get_pp_energy(T, rho, X, Y)
    cno = get_cno_energy(T, rho, X, Y)

    return pp + cno

def load1(P_c, T_c, X, Y):
    """Set up inner boundary conditions given central pressure and temperature

    Parameters
    ----------
    P_c : float
        Central pressure in g cm^-1 s^-2
    T_c : _type_
        Central temperature in K
    X : float
        Hydrogen fraction
    Y : float
        Helium fraction

    Returns
    -------
    list
        Luminosity, pressure, radius, and temperature at the inner boundary
    """
    rho_c = get_density(P_c, T_c, X, Y)
    epsilon_c = get_energy(T_c, rho_c, X, Y)

    # m starts next to 0
    m = 1e-6

    l = epsilon_c * m
    r = (3 * m / (4 * np.pi * rho_c))**(1/3)
    P = P_c - (3 * G / (8 * np.pi)) * (4 * np.pi * rho_c / 3)**(4/3) * m**(2/3)
    T = T_c

    return [l, P, r, T]

def load2(L, R, M):
    """Set up outer boundary conditions given total luminosity and radius

    Parameters
    ----------
    L : float
        Total luminosity in g cm^2 s^-3 (erg/s)
    R : float
        Total radius in cm
    M : float
        Total mass in g

    Returns
    -------
    list
        Luminosity, pressure, radius, and temperature at the outer boundary
    """
    kappa = 0.34

    l = L
    r = R
    T = (L/(4 * np.pi * R**2 * sb))**(1/4)
    P = G * M / r**2 * 2/3 / kappa
    
    return [l, P, r, T]

def get_grad(P, T, kappa, l, m):
    """Calculate radiative gradient and determine whether the radiative or adiabatic regime is appropriate

    Parameters
    ----------
    P : float
        Pressure in g cm^-1 s^-2
    T : float
        Temperature in K
    kappa : float
        Opacity in cm^2 g^-1
    l : float
        Luminosity in g cm^2 s^-3 (erg/s)
    m : float
        Mass in g

    Returns
    -------
    float
        Appropriate gradient value
    """
    grad_ad = 0.4
    grad_rad = (3 / (16 * np.pi * a * c * G)) * ( kappa * l * P / (m * T**4))

    return np.min([grad_ad, grad_rad])

def derivs(m, vals):
    """Calculated derivatives for each ODE using passed m and [l, P, r, T] values

    Parameters
    ----------
    m : float
        Mass in g
    vals : list
        List containing l (in erg/s), P (in g cm^-1 s^-2), r (in cm), and T (in K)

    Returns
    -------
    list
        List of derivatives calculated using values provided
    """
    l, P, r, T = vals

    rho = get_density(P, T, X, Y)
    kappa = get_opacity(T, rho)
    grad = get_grad(P, T, kappa, l, m)

    dldm = get_energy(T, rho, X, Y)
    dPdm = -G * m / (4 * np.pi * r**4)
    drdm = 1/(4 * np.pi * r*2 * rho)
    dTdm = -G * m * T * grad / (4 * np.pi * r**4 * P)

    return [dldm, dPdm, drdm, dTdm]