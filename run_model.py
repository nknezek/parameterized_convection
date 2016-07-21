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

Andrault_f_perioditic = params.Andrault_2011_Stevenson(composition="f_perioditic", Stevenson_case=2)
Andrault_a_chondritic = params.Andrault_2011_Stevenson(composition="a_chondritic", Stevenson_case=2)
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



### Calculate other variables
num_to_plot = 1000
dN = int(len(t)/num_to_plot)
T_cmb = y[::dN,0]
T_mo = y[::dN,1]
T_um = y[::dN,2]
t_plt = t[::dN]/(365.25*24.*3600.*1e6)
r_i = np.array([Earth.core_layer.inner_core_radius(T) for T in T_cmb])

t_all = np.array(Earth.t_all)/(365.25*24.*3600.*1e6)
D = np.array(Earth.D_mo)
T_umo = np.array(Earth.T_umo)
t_all_filtered, D_mo_filtered = Earth.filter_ODE_results(t_all, D)
t_all_filtered, T_umo_filtered = Earth.filter_ODE_results(t_all, T_umo)
T_umo = T_umo[::dN]
D_mo = D_mo_filtered[::dN]
t_filtered = t_all_filtered[::dN]

### Plot Results
peplot.plot_thermal_history(t_plt, T_cmb, T_um, times_calculated=t_filtered, T_upper_magma_ocean=T_umo, D_magma_ocean=D_mo, r_i=r_i)


## Calculate Solubility and Exsolution

P_cmb = 135e6 # Pa
initial_concentration = 0.01
Tin = y[::dN,0]
Pin = P_cmb*np.ones(len(Tin))

# Mg_sol = sol.MgDubrovinskaia()
# deg_fit = 1
# solubility = Mg_sol.solubility(Pin, Tin, deg=deg_fit)
Mg_sol = sol.MgBadro()
solubility = Mg_sol.solubility_OxyRatio(Tin, X_MgO=0.5)

wt = Mg_sol.solubility_to_wt(solubility)
ex_wt = Mg_sol.exsolution(wt, t_plt, initial_concentration=initial_concentration)
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
