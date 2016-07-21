import matplotlib.pyplot as plt
import numpy as np
const_yr_to_sec = 31557600
def plot_thermal_history(times, T_cmb, T_upper_mantle, times_calculated=None, T_upper_magma_ocean=None, D_magma_ocean=None):
    fig = plt.figure(figsize=(8,6))
    plt.plot( times, T_cmb)
    plt.plot( times_calculated, T_upper_magma_ocean)
    plt.plot( times, T_upper_mantle)
    plt.plot( times_calculated, D_magma_ocean/1e2)
    plt.title("Thermal Evolution of Earth")
    plt.ylabel(r"Temperature (K) or thickness (*0.1km)")
    plt.xlabel(r"Time (Myr)")
    plt.legend(["T - Core Mantle Boundary", "T - Upper Magma Ocean", "T - Upper Mantle", "Thickness of Magma Ocean"], loc=0)
    plt.grid()
    plt.savefig("thermal_evolution_T_cmb{0:.0f}K_t{1:.0f}Myr.png".format(T_cmb[0], times[-1]))

def plot_badro_T():
    BadroT = np.recfromcsv('BadroTcmb.csv')
    time_Badro = BadroT['time']
    T_cmb_Badro = BadroT['temperature']

    plt.plot(time_Badro,T_cmb_Badro, 'o')
    plt.grid()
    plt.ylim([4000, 5500])