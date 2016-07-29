# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 22:57:52 2016

@author: nknezek
"""
import numpy as np
import matplotlib.pyplot as plt
import dynamo_library as dyn
from imp import reload
reload(dyn)
const_yr_to_sec = 365.25*24.*3600.

Q_sp = 4.8
Q_Lp = 4.5
Q_gp = 2.7
Q_Rp = 0
Q_kp = 15.0
Q_cmbp = 12.0
E_sp = 144
E_Lp = 258
E_gp = 639
E_Rp = 0
E_kp = -450
Delta_Ep = 591
dT_cmb_dtp = -82 # K/Gyr
dri_dt = 829 # km/Gyr
T_cenp = 5726
T_ip = 5508
T_cmbp = 4180
r_ip = 1220.
C_rp = -10100
dTm_drp = 0.48
dTa_dr_icbp = 0.35
dTa_dr_cmbp = 0.76
P_icbp = 328
P_cmbp = 139

#%%
core = dyn.core_energetics(c_0=0.1, T_cmb0=5500.)
T_cmb = 4179.5
# T_cmb = 4235
r_cmb = 3480e3
dr = 1.
dT_cmb_dt = dT_cmb_dtp/(const_yr_to_sec*1e9) # K/s

print('\ndT_cmb_dt \t= {0:.1f} K/Gyr ({1:.0f} K/Gyr)'.format(dT_cmb_dt*(const_yr_to_sec*1e9), dT_cmb_dtp))
print('T_cmb \t= {0:.1f} K ({1:.0f} K)'.format(T_cmb, T_cmbp))
r_i = core.r_i(T_cmb)
print('\nr_i \t= {0:.1f} km ({1:.0f} km)'.format(r_i/1e3, r_ip))
T_i = core.T_adiabat_from_T_cmb(T_cmb, r_i)
print('T_i \t= {0:.1f} K ({1:.0f} K)'.format(T_i, T_ip))
T_cen = core.T_cen_from_T_cmb(T_cmb)
print('T_cen \t= {0:.1f} K ({1:.0f} K)'.format(T_cen, T_cenp))
C_r = core.C_r(T_cmb)
print('C_r \t= {0:.1f} m/K ({1:.0f} m/K)'.format(C_r, C_rp))
C_c = core.C_c(T_cmb)
print('C_c \t= {0:.1e}'.format(C_c))
dr_i_dt = core.dr_i_dt(T_cmb, dT_cmb_dt)
print('dr_i_dt \t= {0:.1f} km/Gyr'.format(dr_i_dt*const_yr_to_sec*1e9/1e3))
P_icb = core.P(r_cmb)
print('P_icb \t= {0:.1f} GPa ({0:.0f} GPa)'.format(P_icb/1e9, P_icbp))
P_cmb = core.P(r_i)
print('P_cmb \t= {0:.1f} GPa ({0:.0f} GPa)'.format(P_cmb/1e9, P_cmbp))
P_rp = core.P(r_i+dr)
dTm_dr = core.T_m(P_rp) - core.T_m(P_cmb)
print('dTm_dr_icb \t= {0:.2f} K/km ({1:.2f} K/km)'.format(dTm_dr*1e3, -dTm_drp))
dTa_dr_icb = core.T_adiabat_from_T_cmb(T_cmb, r_i+dr)-core.T_adiabat_from_T_cmb(T_cmb, r_i)
print('dTa_dr_icb \t= {0:.2f} K/km ({1:.2f} K/km)'.format(dTa_dr_icb*1e3, -dTa_dr_icbp))
dTa_dr_cmb = core.T_adiabat_from_T_cmb(T_cmb, r_cmb+dr) - core.T_adiabat_from_T_cmb(T_cmb, r_cmb)
print('dTa_dr_cmb \t= {0:.2f} K/km ({1:.2f} K/km)'.format(dTa_dr_cmb*1e3, -dTa_dr_cmbp))

M_c = core.compute_mass_of_core()
M_c2 = core.compute_mass_of_partial_core(r_cmb, 0.)
print('\nM_c \t= {0:.3f} x10^24 kg or {1:.3f} x10^24 kg'.format(M_c/1e24, M_c2/1e24))
M_oc = core.compute_mass_of_partial_core(r_cmb, r_i)
print('M_oc \t= {0:.3f} x10^24 kg'.format(M_oc/1e24))
M_ic = core.compute_mass_of_partial_core(r_i, 0.)
print('M_ic \t= {0:.3f} x10^24 kg'.format(M_ic/1e24))

Q_s = core.Q_s(T_cmb, dT_cmb_dt)
print('\nQ_s \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_s/1e12, Q_sp))
Q_L = core.Q_L(T_cmb, dT_cmb_dt)
print('Q_L \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_L/1e12, Q_Lp))
Q_g = core.Q_g(T_cmb, dT_cmb_dt)
print('Q_g \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_g/1e12, Q_gp))
Q_R = core.Q_R(0.)
print('Q_R \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_R/1e12, Q_Rp))
Q_k = core.Q_k(T_cmb)
print('Q_k \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_k/1e12, Q_kp))
Q_cmb = core.Q_cmb(T_cmb, dT_cmb_dt, 0.)
print('Q_cmb \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_cmb/1e12, Q_cmbp))

Qt_s = core.Qt_s(T_cmb)*dT_cmb_dt
print('\nQt_s \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_s/1e12, Q_sp))
Qt_L = core.Qt_L(T_cmb)*dT_cmb_dt
print('Qt_L \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_L/1e12, Q_Lp))
Qt_g = core.Qt_g(T_cmb)*dT_cmb_dt
print('Qt_g \t= {0:.1f} TW ({1:.1f} TW)'.format(Q_g/1e12, Q_gp))
Et_s = core.Et_s(T_cmb)*dT_cmb_dt
print('Et_s \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(Et_s/1e6, E_sp))
Et_L = core.Et_L(T_cmb)*dT_cmb_dt
print('Et_L \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(Et_L/1e6, E_Lp))
Et_g = core.Et_g(T_cmb)*dT_cmb_dt
print('Et_g \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(Et_g/1e6, E_gp))

E_s = core.E_s(T_cmb, dT_cmb_dt)
print('\nE_s \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(E_s/1e6, E_sp))
E_L = core.E_L(T_cmb, dT_cmb_dt)
print('E_L \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(E_L/1e6, E_Lp))
E_g = core.E_g(T_cmb, dT_cmb_dt)
print('E_g \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(E_g/1e6, E_gp))
E_R = core.E_R(T_cmb, 0.)
print('E_R \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(E_R/1e6, E_Rp))
E_k = core.E_k()
print('E_k \t= {0:.1f} MW/K ({1:.0f} MW/K)'.format(-E_k/1e6, E_kp))
Delta_E = core.Delta_E(T_cmb, dT_cmb_dt, 0.)
print("Del_E \t= {0:.1f} MK/K ({1:.0f} MW/K)".format(Delta_E/1e6, Delta_Ep))

Q_phi = core.Q_phi(T_cmb, dT_cmb_dt, 0., 5000.)
print("\nQ_phi \t= {0:.1f} TW".format(Q_phi/1e12))
E_phi = core.E_phi(T_cmb, dT_cmb_dt, 0.)
print("E_phi \t= {0:.1f} MK/K".format(E_phi/1e6))

D_stable = core.stable_layer_thickness(T_cmb, dT_cmb_dt, 0.)
# A_cmb = 4*np.pi*r_cmb**2
# k=130
# dTq_dr_cmb = -core.Q_cmb(T_cmb, dT_cmb_dt, 0.)/(A_cmb*k)
# print('dTq_dr = {0:.2f} K/km'.format(dTq_dr_cmb*1e3))
# print('dTa_dr = {0:.2f} K/km'.format(dTa_dr_cmb*1e3))
print('D_stable = {0:.1f} km'.format(D_stable/1e3))
print('Q_k = {0:.1f} TW'.format(Q_k/1e12))




r_core = np.linspace(0., r_cmb, 1000)
dTa_dr = core.dTa_dr(T_cmb, r_core)
plt.plot(r_core/1e3, dTa_dr*1e3)
plt.grid()
plt.show()
plot=True
if plot:
    # plot adiabat, inner core radius, and
    r_core = np.linspace(0., r_cmb, 1000)
    P_core = core.P(r_core)
    T_a_core = core.T_adiabat_from_T_cmb(T_cmb, r_core)
    T_m_core = core.T_m(P_core)
    rho_core = core.rho(r_core)
    g_core = core.g(r_core)
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(14,6))
    ax0.plot(r_core/1e3, P_core/1e9, 'b')
    ax0.plot(r_i*np.ones(2)/1e3, np.array([100, 400]), 'k-')
    ax0.set_title('Pressure and Density')
    ax0.set_ylabel('pressure (GPa)', color='b')
    ax0.set_xlabel('radius (km)')
    for tl in ax0.get_yticklabels():
        tl.set_color('b')
    ax0r = ax0.twinx()
    ax0r.plot(r_core/1e3, rho_core, 'g')
    ax0r.set_ylabel('density (kg/m^3)', color='g')
    for tl in ax0r.get_yticklabels():
        tl.set_color('g')
    ax0.grid()

    ax1.plot(r_core/1e3, T_a_core, 'b-')
    ax1.plot(r_core/1e3, T_m_core, 'b--')
    ax1.plot(r_i*np.ones(2)/1e3, np.array([3000, 6000]), 'k-')
    ax1.legend(['Adiabat','Melting', 'inner-core radius'], loc=4)
    ax1.set_title('Temperature and Gravity')
    ax1.set_xlabel('radius (km)')
    ax1.set_ylabel('temperature (K)', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax1.grid()
    ax1r = ax1.twinx()
    ax1r.plot(r_core/1e3, g_core, 'g')
    ax1r.set_ylabel('gravity (m/s^2)', color='g')
    for tl in ax1r.get_yticklabels():
        tl.set_color('g')
    plt.tight_layout(0.1)
    plt.savefig('core_interior_dyanmo_test_{0:.0f}K.pdf'.format(T_cmb))
    # plt.show()
