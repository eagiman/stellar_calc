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

def concat(going_in, going_out):
    L_in, P_in, R_in, T_in = going_in.y
    L_out, P_out, R_out, T_out = going_out.y
    L = np.concatenate([L_out[:-1], L_in[::-1]])
    P = np.concatenate([P_out[:-1], P_in[::-1]])
    R = np.concatenate([R_out[:-1], R_in[::-1]])
    T = np.concatenate([T_out[:-1], T_in[::-1]])
    m = np.concatenate([going_out.t[:-1], going_in.t[::-1]])
    return L, P, R, T, m
    
def make_csv(sol, save=False):
    best_in, best_out = shootf(sol)
    L, P, R, T, m = concat(best_in, best_out)
    
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
        df.to_csv('one_point_five_table.csv', index=False)

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


def plot_quad(sol, save=False, name="plot"):

    best_in, best_out = shootf(sol, fine=True)

    solar_m_in = best_in.t/Ms
    solar_m_out = best_out.t/Ms
    all_in_sol = best_in.y
    all_out_sol = best_out.y

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(14,10))

    # plot l
    ax0.plot(solar_m_in, all_in_sol[0], c=colors[0])
    ax0.plot(solar_m_out, all_out_sol[0], c=colors[0])
    ax0.set_ylabel(r"Luminosity (erg s$^{-1}$)")
    ax0.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax0.axvline(m_int, c="silver", ls="--")

    # plot P
    ax1.plot(solar_m_in, all_in_sol[1], c=colors[1])
    ax1.plot(solar_m_out, all_out_sol[1], c=colors[1])
    ax1.set_ylabel(r"Pressure (dyn cm$^{-2}$)")
    ax1.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax1.axvline(m_int, c="silver", ls="--", label="Fitting Point")

    # plot r
    ax2.plot(solar_m_in, all_in_sol[2], c=colors[2])
    ax2.plot(solar_m_out, all_out_sol[2], c=colors[2])
    ax2.set_ylabel("Radius (cm)")
    ax2.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax2.axvline(m_int, c="silver", ls="--")

    # plot T
    ax3.plot(solar_m_in, all_in_sol[3], c=colors[3])
    ax3.plot(solar_m_out, all_out_sol[3], c=colors[3])
    ax3.set_ylabel("Temperature (K)")
    ax3.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax3.axvline(m_int, c="silver", ls="--")

    ax1.legend()
    
    if save:
        save_as = name + ".png"
        plt.savefig(save_as, dpi=600)
    plt.show() 

def plot_over(sol, initial, save=False, name="over_plot"):

    best_in, best_out = shootf(sol, fine=True)
    init_in, init_out = shootf(initial, fine=True)

    solar_m_in = best_in.t/Ms
    solar_m_out = best_out.t/Ms
    all_in_sol = best_in.y
    all_out_sol = best_out.y

    init_m_in = init_in.t/Ms
    init_m_out = init_out.t/Ms
    init_all_in = init_in.y
    init_all_out = init_out.y

    # Fitting point
    m_int = (M/Ms)/3

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(14,10))

    # plot l
    ax0.plot(solar_m_in, all_in_sol[0], c=colors[0], label="Converged Luminosity")
    ax0.plot(solar_m_out, all_out_sol[0], c=colors[0])
    ax0.plot(init_m_in, init_all_in[0], c=colors[0], alpha=0.8, ls=(0, (1, 1)), label="Initial Luminosity")
    ax0.plot(init_m_out, init_all_out[0], c=colors[0], alpha=0.8, ls=(0, (1, 1)))
    ax0.set_ylabel(r"Luminosity (erg s$^{-1}$)")
    ax0.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax0.axvline(m_int, c="silver", ls="--")

    # plot P
    ax1.plot(solar_m_in, all_in_sol[1], c=colors[1], label="Converged Pressure")
    ax1.plot(solar_m_out, all_out_sol[1], c=colors[1])
    ax1.plot(init_m_in, init_all_in[1], c=colors[1], alpha=0.8, ls=(0, (1, 1)), label="Initial Pressure")
    ax1.plot(init_m_out, init_all_out[1], c=colors[1], alpha=0.8, ls=(0, (1, 1)))
    ax1.set_ylabel(r"Pressure (dyn cm$^{-2}$)")
    ax1.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax1.axvline(m_int, c="silver", ls="--")

    # plot r
    ax2.plot(solar_m_in, all_in_sol[2], c=colors[2], label="Converged Radius")
    ax2.plot(solar_m_out, all_out_sol[2], c=colors[2])
    ax2.plot(init_m_in, init_all_in[2], c=colors[2], alpha=0.8, ls=(0, (1, 1)), label="Initial Radius")
    ax2.plot(init_m_out, init_all_out[2], c=colors[2], alpha=0.8, ls=(0, (1, 1)))
    ax2.set_ylabel("Radius (cm)")
    ax2.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax2.axvline(m_int, c="silver", ls="--")

    # plot T
    ax3.plot(solar_m_in, all_in_sol[3], c=colors[3], label="Converged Temperature")
    ax3.plot(solar_m_out, all_out_sol[3], c=colors[3])
    ax3.plot(init_m_in, init_all_in[3], c=colors[3], alpha=0.8, ls=(0, (1, 1)), label="Initial Temperature")
    ax3.plot(init_m_out, init_all_out[3], c=colors[3], alpha=0.8, ls=(0, (1, 1)))
    ax3.set_ylabel("Temperature (K)")
    ax3.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax3.axvline(m_int, c="silver", ls="--")

   # ax0.legend()
    ax1.legend()
   # ax2.legend()
   # ax3.legend()

    if save:
        save_as = name + ".png"
        plt.savefig(save_as, dpi=600)
    plt.show()

def plot_mesa(sol, p5, save=False, name="mesa_plot"):

    best_in, best_out = shootf(sol, fine=True)

    solar_m_in = best_in.t/Ms
    solar_m_out = best_out.t/Ms
    all_in_sol = best_in.y
    all_out_sol = best_out.y

    # Fitting point
    m_int = (M/Ms)/3

    l = get_mesa_lum(p5)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(14,10))

    # plot l
    ax0.plot(solar_m_in, all_in_sol[0], c=colors[0])
    ax0.plot(solar_m_out, all_out_sol[0], c=colors[0])
    #ax0.plot(p5.mass, init_all_in[0], c=colors[0], alpha=0.4)
    ax0.plot(p5.mass[:-1], l, c=colors[4])
    ax0.set_ylabel(r"Luminosity (erg s$^{-1}$)")
    ax0.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax0.axvline(m_int, c="silver", ls="--")

    # plot P
    ax1.plot(solar_m_in, all_in_sol[1], c=colors[1])
    ax1.plot(solar_m_out, all_out_sol[1], c=colors[1])
    ax1.plot(p5.mass, p5.P, c=colors[4], label="MESA")
    ax1.set_ylabel(r"Pressure (dyn cm$^{-2}$)")
    ax1.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax1.axvline(m_int, c="silver", ls="--")

    # plot r
    ax2.plot(solar_m_in, all_in_sol[2], c=colors[2])
    ax2.plot(solar_m_out, all_out_sol[2], c=colors[2])
    ax2.plot(p5.mass, p5.R*Rs, c=colors[4])
    ax2.set_ylabel("Radius (cm)")
    ax2.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax2.axvline(m_int, c="silver", ls="--")

    # plot T
    ax3.plot(solar_m_in, all_in_sol[3], c=colors[3])
    ax3.plot(solar_m_out, all_out_sol[3], c=colors[3])
    ax3.plot(p5.mass, p5.T, c=colors[4])
    ax3.set_ylabel("Temperature (K)")
    ax3.set_xlabel(r"Mass (M/M$_{\odot})$")
    ax3.axvline(m_int, c="silver", ls="--")

    ax1.legend()

    if save:
        save_as = name + ".png"
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

def plot_grad(solution, save=False):
    sol_in, sol_out = shootf(solution, fine=True)

    m_in, m_in_sol = sol_in.t, sol_in.t/Ms
    m_out, m_out_sol = sol_out.t, sol_out.t/Ms
    m_all = np.concatenate((m_out_sol, m_in_sol))

    L_in, P_in, R_in, T_in = sol_in.y
    L_out, P_out, R_out, T_out = sol_out.y

    rho_in = get_density(P_in, T_in)
    rho_out = get_density(P_out, T_out)

    kappa_in = get_opacity(T_in, rho_in)
    kappa_out = get_opacity(T_out, rho_out)
    
    grad_rad_in = (3 / (16 * np.pi * a * c * G)) * ( kappa_in * L_in * P_in / (m_in * T_in**4))
    grad_rad_out = (3 / (16 * np.pi * a * c * G)) * ( kappa_out * L_out * P_out / (m_out * T_out**4))

    # grad_ad_in = np.ones(len(m_in_sol)) * 0.4
    # grad_ad_out = np.ones(len(m_out_sol)) * 0.4

    grad_ad_in = get_grad_ad(P_in, T_in)
    grad_ad_out = get_grad_ad(P_out, T_out)

    grad_in = get_grad(P_in, T_in, kappa_in, L_in, m_in)
    grad_out = get_grad(P_out, T_out, kappa_out, L_out, m_out)

    plt.figure(figsize=(12,5))

    plt.plot(m_in_sol, grad_rad_in, c="dodgerblue")

    plt.plot(m_out_sol, grad_rad_out, c="dodgerblue", label=r"$\nabla_{\text{rad}}$")
    plt.plot(m_in_sol, grad_ad_in, c="crimson", label=r"$\nabla_{\text{ad}}$")
    plt.plot(m_out_sol, grad_ad_out, c="crimson")
    
    plt.plot(m_in_sol, grad_in, c="black", ls="--", label=r"$\nabla$")
    plt.plot(m_out_sol, grad_out, c="black", ls="--")    
    
    plt.xlabel(r"Mass (M/M$_{\odot}$)")
    plt.ylabel(r"Temperature Gradient $\nabla$")
    plt.legend()

    if save:
        plt.savefig("one_point_five_grad.png", dpi=600)

    plt.show()