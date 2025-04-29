from astropy.io import ascii
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def load_opacity_table(table="top73.txt"):
    """
    Construct an interpolator for the full opacity table

    Parameters
    ----------
    table : str, optional
        file name of opacity table as a .txt file, by default "top73.txt"

    Returns
    -------
    scipy.interpolate._rgi.RegularGridInterpolator
        scipy interpolation object for opacity table
    """

    # Load opacity table and remove placeholders
    table = ascii.read(table, guess=False).to_pandas(index="logT")
    table = table[table != 9.999]

    # Get arrays for interpolation
    indices = table.index.to_numpy().astype(np.float64) # logT
    columns = table.columns.to_numpy().astype(np.float64) # logR
    values = table.to_numpy() # opacities

    # Construct interpolator and return
    return RegularGridInterpolator((indices, columns), values, bounds_error=False, fill_value=None)

# Create interp object
interp = load_opacity_table(table="top73.txt")

def get_opacity(T, rho):
    """
    Interpolate to get opacity

    Parameters
    ----------
    T : float
        Temperature in K
    rho : float
        Density in g cm^-3

    Returns
    -------
    float
        Interpolated opacity value for given T and rho
    """

    # Clip T
    T = np.clip(T, 10**3.5, 10**9)
    T6 = T / 1e6
    R = rho/T6**3
    # Clip R
    R = np.clip(R, 1e-10, 10)

    return 10**interp((np.log10(T), np.log10(R)))