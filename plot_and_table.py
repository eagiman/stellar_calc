from shoot import *
import matplotlib.pyplot as plt
from constants import *
import pandas as pd
import numpy as np
import scienceplots
plt.style.use(["science", "notebook"])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

M = 1.5 * Ms

# Intermediary point in solar mass
m_int = (M/Ms)/3

def concat(guess, fine=False):

    going_in, going_out = shootf(guess, fine=fine)

    L_in, P_in, R_in, T_in = going_in.y
    L_out, P_out, R_out, T_out = going_out.y
    L = np.concatenate([L_out[:-1], L_in[::-1]])
    P = np.concatenate([P_out[:-1], P_in[::-1]])
    R = np.concatenate([R_out[:-1], R_in[::-1]])
    T = np.concatenate([T_out[:-1], T_in[::-1]])
    m = np.concatenate([going_out.t[:-1], going_in.t[::-1]])

    return L, P, R, T, m
    
def make_csv(sol, save=False):

    L, P, R, T, m = concat(sol)
    
    dic = {
        "m": m,
        "L": L,
        "P": P,
        "R": R,
        "T": T
    }
    df = pd.DataFrame(dic)
    
    df["density"] = get_density(P, T)
    df["energy"] = get_energy(T, df["density"])
    df["pp"] = get_pp(T, df["density"])
    df["cno"] = get_cno(T, df["density"])
    df["opacity"] = get_opacity(T, df["density"])
    #df["adiabatic gradient"] = np.ones(len(m)) * 0.4
    df["adiabatic gradient"] = get_grad_ad(P, T)
    df["radiative gradient"] = get_grad_rad(P, T, df["opacity"], L, m)
    df["actual gradient"] = np.where(df["adiabatic gradient"] < df["radiative gradient"], df["adiabatic gradient"], df["radiative gradient"])
    df["regime"] = np.where(df["adiabatic gradient"] < df["radiative gradient"], "convective", "radiative")
    
    if save:
        df.to_csv('results.csv', index=False)

    return df

def get_mesa_lum(p5):
    energy = p5.pp + p5.cno
    energy_order = energy[::-1]
    mass_order = p5.mass[::-1]
    dm = np.diff(mass_order)
    energy_mid = (energy_order[:-1] + energy_order[1:])/2
    dl = energy_mid * dm
    l = np.cumsum(dl)
    l = l[::-1] * Ls
    return l


def plot(sol, name=None):

    L, P, R, T, m_g = concat(sol, fine=True)

    m = m_g/Ms

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(14,10))

    # plot l
    ax0.plot(m, L, c=colors[0])
    ax0.set_ylabel(r"Luminosity (erg s$^{-1}$)")
    ax0.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax0.axvline(m_int, c="silver", ls="--")

    # plot P
    ax1.plot(m, P, c=colors[1])
    ax1.set_ylabel(r"Pressure (dyn cm$^{-2}$)")
    ax1.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax1.axvline(m_int, c="silver", ls="--", label="Fitting Point")

    # plot r
    ax2.plot(m, R, c=colors[2])
    ax2.set_ylabel("Radius (cm)")
    ax2.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax2.axvline(m_int, c="silver", ls="--")

    # plot T
    ax3.plot(m, T, c=colors[3])
    ax3.set_ylabel("Temperature (K)")
    ax3.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax3.axvline(m_int, c="silver", ls="--")

    ax1.legend()
    
    if name is not None:
        save_as = name + ".pdf"
        plt.savefig(save_as, dpi=600)
    plt.show() 

def plot_over(sol, initial, name=None):

    L, P, R, T, m_g = concat(sol, fine=True)
    L_init, P_init, R_init, T_init, m_g_init = concat(initial, fine=True)
    m = m_g/Ms

    # Fitting point
    m_int = (M/Ms)/3

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(14,10))

    # plot l
    ax0.plot(m, L, c=colors[0], label="Converged Luminosity")
    ax0.plot(m, L_init, c=colors[0], alpha=0.8, ls=(0, (1, 1)), label="Initial Luminosity")
    ax0.set_ylabel(r"Luminosity (erg s$^{-1}$)")
    ax0.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax0.axvline(m_int, c="silver", ls="--")

    # plot P
    ax1.plot(m, P, c=colors[1], label="Converged Pressure")
    ax1.plot(m, P_init, c=colors[1], alpha=0.8, ls=(0, (1, 1)), label="Initial Pressure")
    ax1.set_ylabel(r"Pressure (dyn cm$^{-2}$)")
    ax1.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax1.axvline(m_int, c="silver", ls="--")

    # plot r
    ax2.plot(m, R, c=colors[2], label="Converged Radius")
    ax2.plot(m, R_init, c=colors[2], alpha=0.8, ls=(0, (1, 1)), label="Initial Radius")
    ax2.set_ylabel("Radius (cm)")
    ax2.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax2.axvline(m_int, c="silver", ls="--")

    # plot T
    ax3.plot(m, T, c=colors[3], label="Converged Temperature")
    ax3.plot(m, T_init, c=colors[3], alpha=0.8, ls=(0, (1, 1)), label="Initial Temperature")
    ax3.set_ylabel("Temperature (K)")
    ax3.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax3.axvline(m_int, c="silver", ls="--")

   # ax0.legend()
    ax1.legend()
   # ax2.legend()
   # ax3.legend()

    if name is not None:
        save_as = name + ".pdf"
        plt.savefig(save_as, dpi=600)
    plt.show()

def plot_mesa(sol, p5, name=None):

    L, P, R, T, m_g = concat(sol, fine=True)
    m = m_g/Ms

    # Fitting point
    m_int = (M/Ms)/3

    l = get_mesa_lum(p5)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(14,10))

    # plot l
    ax0.plot(m, L, c=colors[0])
    ax0.plot(p5.mass[:-1], l, c=colors[4])
    ax0.set_ylabel(r"Luminosity (erg s$^{-1}$)")
    ax0.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax0.axvline(m_int, c="silver", ls="--")

    # plot P
    ax1.plot(m, P, c=colors[1])
    ax1.plot(p5.mass, p5.P, c=colors[4], label="MESA")
    ax1.set_ylabel(r"Pressure (dyn cm$^{-2}$)")
    ax1.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax1.axvline(m_int, c="silver", ls="--")

    # plot r
    ax2.plot(m, R, c=colors[2])
    ax2.plot(p5.mass, p5.R*Rs, c=colors[4])
    ax2.set_ylabel("Radius (cm)")
    ax2.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax2.axvline(m_int, c="silver", ls="--")

    # plot T
    ax3.plot(m, T, c=colors[3])
    ax3.plot(p5.mass, p5.T, c=colors[4])
    ax3.set_ylabel("Temperature (K)")
    ax3.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax3.axvline(m_int, c="silver", ls="--")

    ax1.legend()

    if name is not None:
        save_as = name + ".pdf"
        plt.savefig(save_as, dpi=600)
    plt.show()

def error(measured, true):
        return np.abs(measured - true)/true

def frac_diffs(sol, p5):

    best_in, best_out = shootf(sol, fine=True)

    l = get_mesa_lum(p5)
    my_L = sol[0]
    mesa_L = l[0]

    my_P = sol[1]
    mesa_P = p5.P[-1]

    my_R = sol[2]
    mesa_R = p5.R[0] * Rs

    my_T_c = sol[3]
    mesa_T_c = p5.T[-1]

    my_T_eff = best_in.y[3][0]
    mesa_T_eff = p5.T[0]

    my_g = G*M / my_R**2
    mesa_g = G*M / mesa_R**2

    L_error = error(my_L, mesa_L)
    P_error = error(my_P, mesa_P)
    R_error = error(my_R, mesa_R)
    T_c_error = error(my_T_c, mesa_T_c)
    T_eff_error = error(my_T_eff, mesa_T_eff)
    g_error = error(my_g, mesa_g)

    print(f"L error: {L_error}")
    print(f"P_c error: {P_error}")
    print(f"R error: {R_error}")
    print(f"T_c error: {T_c_error}")
    print(f"T_eff error: {T_eff_error}")
    print(f"g error: {g_error}")

def plot_grad(sol, name=None):

    L, P, R, T, m_g = concat(sol, fine=True)
    m = m_g/Ms

    rho = get_density(P, T)
    kappa = get_opacity(T, rho)

    grad_rad = get_grad_rad(P, T, kappa, L, m_g)
    grad_ad = get_grad_ad(P, T)
    grad = get_grad(P, T, kappa, L, m_g)
    

    plt.figure(figsize=(12,5))


    plt.plot(m, grad_rad, c="dodgerblue", label=r"$\nabla_{\text{rad}}$")
    plt.plot(m, grad_ad, c="crimson", label=r"$\nabla_{\text{ad}}$")
    plt.plot(m, grad, c="black", ls="dashdot", label=r"$\nabla$")
    
    plt.xlabel(r"Mass (M/M$_{\odot}$)")
    plt.ylabel(r"Temperature Gradient $\nabla$")
    plt.legend()

    if name is not None:
        save_as = name + ".pdf"
        plt.savefig(save_as, dpi=600)

    plt.show()