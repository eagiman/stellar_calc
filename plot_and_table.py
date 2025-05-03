from shoot import *
import matplotlib.pyplot as plt
from constants import *
import pandas as pd
import numpy as np
import scienceplots
plt.style.use(["science", "notebook"])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_quad(sol, save=False, name="plot"):

    best_in, best_out = shootf(sol, fine=True)

    solar_m_in = best_in.t/Ms
    solar_m_out = best_out.t/Ms
    all_in_sol = best_in.y
    all_out_sol = best_out.y

    # Intermediary point in solar mass
    m_int = 1.1/3

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
    ax1.axvline(m_int, c="silver", ls="--", label="Intermediary Point")

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

def concat(going_in, going_out):
    L_in, P_in, R_in, T_in = going_in.y
    L_out, P_out, R_out, T_out = going_out.y
    L = np.concatenate([L_out[:-1], L_in[::-1]])
    P = np.concatenate([P_out[:-1], P_in[::-1]])
    R = np.concatenate([R_out[:-1], R_in[::-1]])
    T = np.concatenate([T_out[:-1], T_in[::-1]])
    m = np.concatenate([going_out.t[:-1], going_in.t[::-1]])
    return L, P, R, T, m
    
def make_csv(sol):
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
    df["opacity"] = get_opacity(T, df["density"])
    df["adiabatic gradient"] = np.ones(len(m)) * 0.4
    df["radiative gradient"] = grad_rad(P, T, df["opacity"], L, m)
    df["actual gradient"] = np.where(df["adiabatic gradient"] < df["radiative gradient"], df["adiabatic gradient"], df["radiative gradient"])
    df["regime"] = np.where(df["adiabatic gradient"] < df["radiative gradient"], "convective", "radiative")
    
    df.to_csv('machine_readable_table.csv', index=False)

    return df