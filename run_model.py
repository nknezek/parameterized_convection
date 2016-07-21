import numpy as np
import matplotlib.pyplot as plt
import input_parameters as params
import solubility_library as sol
import plotting_library as peplot
import planetary_energetics as pe
import isotope_library as iso
const_yr_to_sec = 31557600


### Import parameter set choices

Stevenson_E1 = params.Stevenson_1983(case=1)
Stevenson_E2 = params.Stevenson_1983(case=2)
Driscoll = params.Driscoll_2014()
param2layer = Driscoll
# Earth = Planet( [ CoreLayer( 0.0, param2layer.R_c0, params=param2layer) , MantleLayer( param2layer.R_c0, param2layer.R_p0, params=param2layer) ] )

Andrault_f_perioditic = params.Andrault_2011_Stevenson(composition="f_perioditic", Stevenson_case=1)
Andrault_a_chondritic = params.Andrault_2011_Stevenson(composition="a_chondritic", Stevenson_case=1)
# param3layer = Andrault_f_perioditic
param3layer = Andrault_a_chondritic


### Define Earth

Earth = pe.Planet( [ pe.CoreLayer( 0.0, param3layer.R_c0, params=param3layer) ,
                  pe.MagmaOceanLayer(param3layer.R_c0, param3layer.R_mo0, params=param3layer),
                  pe.MantleLayer(param3layer.R_mo0, param3layer.R_p0, params=param3layer) ] )


### Define Initial Conditions

# T_cmb_initial = 8200. # K
# T_magma_ocean_initial = 7245. # K
# T_mantle_initial = 4200. # K
T_cmb_initial = 5500. # K
# T_magma_ocean_initial = 8400. # K
# T_mantle_initial = 5200. # K



### Integrate Model for Solution

end_time_Mya = 4568 # Mya
# end_time_Mya = 14 # Mya
end_time = end_time_Mya*1e6*const_yr_to_sec # s
Nt = 2000*end_time_Mya
times = np.linspace(0., end_time, Nt)
t, y = Earth.integrate(times, T_cmb_initial, None, None, verbose=False)



### Plot Results
num_to_plot = 1000
dN = int(len(t)/num_to_plot)
t_plt = t[::dN]/(365.25*24.*3600.*1e6)
t_all = np.array(Earth.t_all)/(365.25*24.*3600.*1e6)
D = np.array(Earth.D_mo)
T_umo = np.array(Earth.T_umo)
tD, Do = Earth.filter_ODE_results(t_all, D)
t2, T_umo = Earth.filter_ODE_results(t_all, T_umo)
T_umo = T_umo[::dN]
tD = tD[::dN]
Do2 = Do[::dN]
t_all = t_all[::dN]
peplot.plot_thermal_history(t_plt, y[::dN,0], y[::dN,2], times_calculated=tD, T_upper_magma_ocean=T_umo, D_magma_ocean=Do2)

## Calculate Solubility and Exsolution
P_cmb = 135e6 # Pa
initial_concentration = 0.01
deg_fit = 1
# Mg_sol = sol.MgDubrovinskaia()
Mg_sol = sol.MgBadro()
Tin = y[::dN,0]
Pin = P_cmb*np.ones(len(Tin))
# solubility = Mg_sol.solubility(Pin, Tin, deg=deg_fit)
solubility = Mg_sol.solubility(Pin, Tin)
wt = Mg_sol.solubility_to_wt(solubility)
ex_wt = Mg_sol.exsolution(wt, t_plt, initial_concentration=initial_concentration)
# ex_wt = Mg_sol.solubility_to_wt(exsolution)
fig = plt.figure(figsize=(8,6))
plt.plot(t_plt, wt)
plt.plot(t_plt[1:], -ex_wt*1000)
plt.xlabel("Time (Mya)")
plt.ylabel("Mg Solubility (wt%) or Exsolution Rate (wt%/Bya)")
plt.grid()
plt.title("Mg Solubility over time, T_cmb0={0:.0f}K".format(T_cmb_initial))
plt.legend(["Mg Solubility", "Mg Exsolution Rate"])
plt.savefig('Mg_sol_ex.png')


## Calculate Isotopic ratios
a = 0.
b = +0.003 # factor = a+b/T**2
moles = 100.
initial_core_isotope_ratio = 0.999
Mg = iso.MgReservoir(a, b, moles, initial_core_isotope_ratio)
