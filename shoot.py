from constants import *
from structure import *
from scipy.integrate import solve_ivp
from scipy.optimize import root

M = 1.1 * Ms

# Initial guesses

# Homology
R = ((M/Ms)**(0.75)) * Rs 
L = ((M/Ms)**(3.5))*Ls 

# Constant Density
P_c = (3/(8*np.pi))*(G*(M)**2)/(R)**4 
T_c = (((1/2)*mu)/(na*k))*(G*M)/(R) 

#initial_guess = [L, P_c, R, T_c]

# Using sun fact sheet from NASA for P_c and T_c guess
initial_guess = [L, 2.477e17, R, 1.571e7]

def shootf(guess, fine=False):
    """
    Shoot for solution at intermediary value (a la numerical recipes' shootf)

    Parameters
    ----------
    guess : array/list
        List of values (surface L, central P, surface R, central T) to shoot with
    fine : bool, optional
        Whether to track values for very smooth plot, by default False

    Returns
    -------
    sol_in : scipy OptimizeResult
        Result of inward integration
    sol_out : scipy OptimizeResult
        Result of outward integration
    """

    L, P_c, R, T_c = guess

    # Boundary values
    inner_bound = load1(P_c, T_c)
    outer_bound = load2(L, R, M)

    # Intermediary mass
    m_int = M / 3

    if fine:
        steps_in = np.linspace(M, m_int, 1000)
        steps_out = np.linspace(1e-6, m_int, 500)
    else:
        steps_in, steps_out = None, None

    # Integrate inward
    sol_in = solve_ivp(
        derivs,
        [M, m_int],
        outer_bound,
        dense_output = True,
        t_eval = steps_in
    )

    # Integrate outward
    sol_out = solve_ivp(
        derivs,
        [1e-6, m_int],
        inner_bound,
        dense_output = True,
        t_eval = steps_out
    )

    return sol_in, sol_out

def match(guess):
    """
    Determine how far apart the inward and outward integrations were at intermediary point

    Parameters
    ----------
    guess : array/list
         List of values (surface L, central P, surface R, central T) to evaluate

    Returns
    -------
    list
        List of "residuals", i.e. normalized differences between inward and outward integrations
    """

    # Check if guess is numpy array (otherwise just list)
    # If it's not, make it into numpy array for next step
    if not isinstance(guess, np.ndarray):
        guess = np.array(guess)
    
    # Discourage solver if doing weird things
    if any(guess<=0):
        return np.ones(4) * 1e20
    try:
        sol_in, sol_out = shootf(guess)
        vals_in = sol_in.y[:, -1]
        vals_out = sol_out.y[:, -1]
    except:
        return np.ones(4) * 1e10
    
    # Normalized difference at intermediate point
    scale = (vals_in + vals_out) / 2
    residuals = (vals_in - vals_out) / scale
    return residuals

def newt(initial, print_sol=False):
    """
    Run root iteratively to get converging values (based very very loosely on newt from numerical recipes)

    Parameters
    ----------
    initial : list/array
        Initial values (surface L, central P, surface R, central T) to start with

    Returns
    -------
    array
        Converging input values
    """
    status = 0
    runs = 0
    guess = initial
    while status != 1:
        trying = root(match, guess)
        status = trying.status
        runs += 1
        print(f"Run {runs} status: {status}")
        if all(trying.x==guess):
            print("Not going anywhere :(")
            break
        guess = trying.x

    if status==1:
        print(f"Success! Converged in {runs} runs of root.")
    if print_sol==True:
        print(trying)
    return trying.x