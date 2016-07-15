import numpy as np

# ------------------------------------------------------ #
# - Parameters taken from Stevenson 1983 - #
# ------------------------------------------------------ #

# Table I
alpha = 2e4 # - K
k = 4.0 # - W/m-K
K = 1e6 # - m^2/s
rho_mantle_C_mantle = 4.0e6 # - J/m^3 K
rho_core_C_core = rho_mantle_C_mantle
Q_0 = 1.7e7 # W m^3
Q_decay = 1.38e-17
viscosity_0 = 4.0e3 # - m^2/s
viscosity_A = 5.2e4 # - K
Ra_crit = 5.0e2
beta = 0.3

# Table II
radius_earth = 6371e3 # - m
g = 10. # - m/s^2
Ts = 293. # - K
Tm1 = 6.14e12 # - K Pa
Tm2 = -4.5e24 # - K Pa^2
Ta1 = 3.96e12 # - K Pa
Ta2 = -3.3e24 # - K Pa^2
mu_m = 1.3
mu_a = 1.2

# Table III

rho_core = 1.3e4 # - kg/m^3
radius_core = 3485e3 # - m
x0 = 0.1 # wt% S
P_center = 0.14e12 # - Pa


# E1
L_Eg = 1e6 # - J/kg
Tm0 = 1950. # - K
q = -0.6
gamma = rho_core**q
delta_c = 20.
onset = 2.7 # Byrs
Ri = 1234e3 # - m
Fc = 18.6e-3 # - W/m^2
Fs = 62.7e-3 # - W/m^2
Tu = 1648. # - K
Tcm = 2960. # - K
Tmio = 3956.
dRidt = 0.25 # - m/Mya
nu_m = 2e17 # - m^2/s
Ra = 6e8

# E2
L_Eg = 2e6 # - J/kg
Tm0 = 1980. # - K
q = -0.6
gamma = rho_core**q
delta_c = 20.
onset = 2.7 # Byrs
Ri = 1234e3 # - m
Fc = 18.6e-3 # - W/m^2
Fs = 62.7e-3 # - W/m^2
Tu = 1648. # - K
Tcm = 2960. # - K
Tmio = 3956.
dRidt = 0.25 # - m/Mya
nu_m = 2e17 # - m^2/s
Ra = 6e8



core_params = {
    'rho' : rho_core,
    'c'   : core_heat_capacity,
    'L+Eg': latent_heat_solidification_core+grav_energy_release,
    'mu' : epsilon_core
    }

mantle_params = {
    'rho' : rho_mantle,
    'c'   : mantle_heat_capacity,
    'epsilon' : epsilon_mantle
    }

