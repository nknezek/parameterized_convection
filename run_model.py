import numpy as np
import matplotlib.pyplot as plt
import input_parameters as params
import solubility_library as sol
import plotting_library as peplot
import planetary_energetics as pe
import isotope_library as iso
import dynamo_library as dyn
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
param3layer.core = params.Nimmo_2015()


### Define Earth

Earth = pe.Planet( [ dyn.core_energetics( 0.0, param3layer.R_c0, params=param3layer) ,
                  pe.MagmaOceanLayer(param3layer.R_c0, param3layer.R_mo0, params=param3layer),
                  pe.MantleLayer(param3layer.R_mo0, param3layer.R_p0, params=param3layer) ] )


### Define Initial Conditions

# T_cmb_initial = 8200. # K
# T_magma_ocean_initial = 7245. # K
# T_mantle_initial = 4200. # K
T_cmb_initial = 5200. # K
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
t_sec = t[::dN]
t_plt = t_sec/(const_yr_to_sec*1e6)
r_i = np.array([Earth.core_layer.r_i(T, recompute=True, store_computed=False) for T in T_cmb])

t_all = np.array(Earth.t_all)/(365.25*24.*3600.*1e6)
D = np.array(Earth.D_mo)
T_umo = np.array(Earth.T_umo)
t_all_filtered, D_mo_filtered = Earth.filter_ODE_results(t_all, D)
t_all_filtered, T_umo_filtered = Earth.filter_ODE_results(t_all, T_umo)
T_umo = T_umo[::dN]
D_mo = D_mo_filtered[::dN]
t_filtered = t_all_filtered[::dN]

### Plot Results
T_i = Earth.core_layer.T_adiabat_from_T_cmb(T_cmb, r_i)
peplot.plot_thermal_history2(t_plt, T_cmb, T_um, times_calculated=t_filtered, T_upper_magma_ocean=T_umo, D_magma_ocean=D_mo, r_i=r_i, T_i=T_i)


### Plot viscosity
viscosity = Earth.mantle_layer.kinematic_viscosity(T_um)
plt.figure(figsize=(8,6))
plt.semilogy(t_plt, viscosity)
plt.xlabel('time (MYa)')
plt.ylabel('kinematic viscosity (m^2/s)')
# plt.ylim([0,np.max(viscosity)*1.1])
plt.title('viscosity of the mantle over time')
plt.grid()
plt.savefig('viscosity.pdf')


## Calculate Solubility and Exsolution
P_cmb = 135e6 # Pa
# initial_concentration = 0.0095
initial_concentration = 0.0115
Tin = y[::dN,0]
Pin = P_cmb*np.ones(len(Tin))

# Mg_sol = sol.MgDubrovinskaia()
# deg_fit = 1
# solubility = Mg_sol.solubility(Pin, Tin, deg=deg_fit)
Mg_sol = sol.MgBadro()
solubility05 = Mg_sol.solubility_OxyRatio(Tin, X_MgO=0.5)
wt05 = Mg_sol.solubility_to_wt(solubility05)
ex_wt05 = Mg_sol.exsolution(wt05, t_plt, initial_concentration=initial_concentration)

solubility10 = Mg_sol.solubility_OxyRatio(Tin, X_MgO=1.0)
wt10 = Mg_sol.solubility_to_wt(solubility10)
ex_wt10 = Mg_sol.exsolution(wt10, t_plt, initial_concentration=initial_concentration)

fig = plt.figure(figsize=(8,6))
plt.plot(t_plt, wt05*1e2*1.67, 'r-')
plt.plot(t_plt, wt10*1e2*1.67, 'b-')
plt.plot(t_plt[1:], -ex_wt05*1e2*1e3*1.67, 'r--')
plt.plot(t_plt[1:], -ex_wt10*1e2*1e3*1.67, 'b--')
plt.plot(t_plt, np.ones_like(t_plt)*initial_concentration*1e2*1.67, 'k--')
plt.xlabel("Time (Mya)")
plt.ylabel("Mg Solubility (wt%) or Exsolution Rate (wt%/Bya)")
# plt.grid()
plt.title("MgO Solubility and Exsolution".format(T_cmb_initial))
plt.legend(["MgO solubility - magma ocean", "MgO solubility - MgO layer", "MgO exsolution rate - magma ocean", "MgO exsolution rate - MgO layer", "Initial MgO in core"])
plt.savefig('Mg_sol_ex_Tcmb0_{0:.0f}K.pdf'.format(T_cmb_initial))


## Calculate Dynamo Power

T_D = 5000. # K - from Nimmo 2015
dT_cmb_dt = np.diff(T_cmb)/np.diff(t_sec)
Q_phi = np.zeros_like(dT_cmb_dt)
Q_cmb = np.zeros_like(dT_cmb_dt)
E_phi = np.zeros_like(dT_cmb_dt)
E_k = np.zeros_like(dT_cmb_dt)
Q_k = np.zeros_like(dT_cmb_dt)
h = Earth.core_layer.heat_production_per_kg(t_sec)
for i, (T, dT, ht) in enumerate(zip(T_cmb[:-1], dT_cmb_dt, h[:-1])):
    Earth.core_layer.reset_current_values()
    Q_phi[i] = Earth.core_layer.Q_phi(T, dT, ht, T_D)
    Q_cmb[i] = Earth.core_layer.Q_cmb(T, dT, ht)
    Q_k[i] = Earth.core_layer.Q_k(T)
    E_phi[i] = Earth.core_layer.E_phi(T,dT, ht)
    E_k[i] = Earth.core_layer.E_k()

min_E = 200e6 # W/K
min_Q = min_E*T_D # W
fig, (ax0, ax1) = plt.subplots(1,2,figsize=(14,6))
ax0.plot(t_plt[:-1], Q_phi/1e12, 'b-')
ax0.plot(t_plt[:-1], Q_cmb/1e12, 'b-.')
ax0.plot(t_plt[:-1], Q_k/1e12, 'b--')
ax0.plot(t_plt, np.ones_like(t_plt)*min_Q/1e12, color='0.5', alpha=0.5, linewidth=5.)
ax0.grid()
ax0.legend(['Q_phi','Q_cmb','Q_k', 'min_Q'])
ax0.set_ylabel('TW', color='b')
ax0.set_ylim([-5, 30])
for tl in ax0.get_yticklabels():
    tl.set_color('b')
ax1.plot(t_plt[:-1], E_phi/1e6, 'g-')
ax1.plot(t_plt[:-1], E_k/1e6, 'g--')
ax1.plot(t_plt, np.ones_like(t_plt)*min_E/1e6, color='0.5', alpha=0.5, linewidth=5.)
ax1.set_ylabel('MW/K', color='g')
for tl in ax1.get_yticklabels():
    tl.set_color('g')
ax1.grid()
ax1.legend(['E_phi','E_k','min_E'])
plt.savefig('Dynamo_Power_test_Tcmb0_{0:.0f}K.pdf'.format(T_cmb_initial))

# rho_c = 11000
# delta_rho_c = 500
# rho_MgO = rho_c-delta_rho_c
# g = 5
# h = 3480e3/2
# C_p = 800
# L = 1e6
# M_c = Earth.core_layer.volume*rho_c
# dTdt = np.diff(T_cmb)/np.diff(t_sec)
# Q_s = -M_c*C_p*dTdt*5e-2
# Q_g = -delta_rho_c*g*h*ex_wt05/2e16
# Q_L = -rho_c*L*ex_wt05*M_c/2e16
# Q_MgO = Q_g + Q_L
# Q_g2 = -delta_rho_c*g*h*ex_wt10/2e16
# Q_L2 = -rho_c*L*ex_wt10*M_c/2e16
# Q_MgO2 = Q_g2 + Q_L2
#
#
# L_Eg = 1e6
# dRi_dTcmb = np.diff(r_i)/np.diff(T_cmb)
# inner_core_surface_area = 4*np.pi*r_i[:-1]**2
# Q_ic = -L_Eg * rho_c * inner_core_surface_area * dRi_dTcmb/1e15
# Q_total = Q_s+Q_MgO+Q_ic
# Q_total2 = Q_s+Q_MgO2+Q_ic
#
# print(Q_s, Q_g, Q_L)
# fig = plt.figure(figsize=(8,6))
# plt.plot(t_plt[:-1], Q_s/1e12)
# plt.plot(t_plt[:-1], Q_MgO/1e12)
# plt.plot(t_plt[:-1], Q_ic/1e12)
# plt.plot(t_plt[:-1], Q_total/1e12)
# plt.plot(t_plt[:-1], Q_total2/1e12)
# plt.plot(t_plt, np.ones_like(t_plt)*0.9, '-', alpha=0.5, linewidth=10., color='0.5')
# plt.legend(['Q_s','Q_MgO','Q_ic','Q_total'])
# plt.title('Dynamo Power Over Time')
# plt.legend(['magma ocean','MgO layer','Minimum dynamo power'])
# plt.xlabel('Time (Myr)')
# plt.ylabel('Power (TW)')
# plt.xlim([0,4500])
# plt.savefig('Dynamo_Power_test.pdf')


## Calculate Isotopic ratios
a = 0.
b = +0.003 # factor = a+b/T**2
moles = 100.
initial_core_isotope_ratio = 0.999
Mg = iso.MgReservoir(a, b, moles, initial_core_isotope_ratio)
