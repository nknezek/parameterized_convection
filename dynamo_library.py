import input_parameters as params
import numpy as np
import scipy.special as spec
from numpy import pi, exp
import scipy.optimize as opt

class Nimmo_2015(params.Parameters):
    def __init__(self, source):
        super(Nimmo_2015,self).__init__(source)
        self.rho_cen = 12500 # kg / m^3
        self.rho_0 = 7900 # kg/m^3

        self.r_c = 3480e3 # m
        self.r_i = 1220e3 # m

        self.K_0 = 500e9 # Pa
        self.L = 7272e3 # m

        self.P_c = 139e9 # Pa
        self.P_icb = 328e9 # Pa

        self.T_c = 4180 # K
        self.T_i = 5508 # K
        self.T_cen = 5726 # K
        self.T_m0 = 2677 # K
        self.T_m1 = 2.95e-12 # /Pa
        self.T_m2 = 8.37e-25 # /Pa^2

        self.alpha = 1.25e-5 # /K
        self.L_H = 750e3 # J/kg
        self.k = 130 # W/m-K
        self.D = 6203e3 # m
        self.D_k = 5900e3 # pg. 42
        self.C_p = 840 # J/kg-K
        self.alpha_c = 1.1 # -
        self.delta_rho_c = 560 # kg/m^3
        self.C_r = -10100 # m/K
        self.G = 6.67408e-11 # m^3/kg-s

p = Nimmo_2015('Nimmo 2015')

class core_energetics():
    def __init__(self, c_0, T_cmb0):
        self.c = c_0
        self.T_i = 0.
        self.T_cmb = T_cmb0
        self.compute_mass_of_core()

    def T_adiabat_from_T_cen(self, T_cen, r):
        return T_cen*exp(-r**2/p.D**2)

    def T_adiabat_from_T_cmb(self, T_cmb, r):
        T_cen = self.T_cen_from_T_cmb(T_cmb)
        return T_cen*exp(-r**2/p.D**2)

    def T_cen_from_T_cmb(self, T_cmb):
        return T_cmb*exp(p.r_c**2/p.D**2)

    def compute_mass_of_core(self):
        self.mass = 4/3*pi*p.rho_cen *p.r_c**3 * exp(-p.r_c**2/p.L**2) * (1+ 2/5 * p.r_c**2/p.L**2)
        return self.mass

    def compute_mass_of_partial_core(self, r_top, r_bottom):
        return 4*pi*p.rho_cen*((-p.L**2/2*r_top * exp(-r_top**2/p.L**2) + p.L**3/4*pi**0.5*spec.erf(r_top/p.L))
                           -(-p.L**2/2*r_bottom * exp(-r_bottom**2/p.L**2) + p.L**3/4*pi**0.5*spec.erf(r_bottom/p.L)))

    def rho(self, r):
        return p.rho_cen*exp(-r**2/p.L**2)

    def g(self, r):
        return 4*pi/3*p.G*p.rho_cen * r * (1- 3*r**2/(5*p.L**2))

    def P(self, r):
        return p.P_c+4*pi*p.G*p.rho_cen**2/3 * ((3*p.r_c**2/10 - p.L**2/5)*exp(-p.r_c**2/p.L**2) - (3*r**2/10 - p.L**2/5)*exp(-r**2/p.L**2))

    def T_m(self, P):
        return p.T_m0*(1 + p.T_m1*P + p.T_m2*P**2)

    def I_s(self, T_cmb):
        T_cen = self.T_cen_from_T_cmb(T_cmb)
        Asq = (1/p.L**2 + 1/p.D**2)**-1
        return  T_cen *4/3 * pi * p.rho_cen * p.r_c**3 * exp(-p.r_c**2/Asq) * (1 + 2*p.r_c**2 / (5 * Asq))

    def Q_s(self, T_cmb, dT_cmb_dt, I_s=None):
        if I_s is None:
            I_s = self.I_s(T_cmb)
        return -p.C_p/T_cmb*dT_cmb_dt*I_s

    def Qt_s(self, T_cmb, I_s=None):
        if I_s is None:
            I_s = self.I_s(T_cmb)
        return -p.C_p/T_cmb*I_s

    def E_s(self, T_cmb, dT_cmb_dt, I_s=None):
        if I_s is None:
            I_s = self.I_s(self.T_cen_from_T_cmb(T_cmb))
        return p.C_p/T_cmb * (self.mass-I_s/T_cmb)*dT_cmb_dt

    def Et_s(self, T_cmb, I_s=None):
        if I_s is None:
            I_s = self.I_s(self.T_cen_from_T_cmb(T_cmb))
        return p.C_p/T_cmb * (self.mass-I_s/T_cmb)

    def I_T(self, T_cmb):
        T_cen = self.T_cen_from_T_cmb(T_cmb)
        Bsq = (1/p.L**2 - 1/p.D**2)**-1
        return 4*pi*p.rho_cen/(3*T_cen) * p.r_c**3 * (1-3*p.r_c**2/(5*Bsq))

    def Q_R(self, h):
        return self.mass*h

    def E_R(self, h, T_cmb):
        I_T = self.I_T(T_cmb)
        return (self.mass/T_cmb - I_T)*h

    def dr_i_dt(self, dT_cmb_dt):
        return p.C_r*dT_cmb_dt

    def Dc_Dt(self, dT_cmb_dt, r_i):
        return self.C_c(r_i)*p.C_r*dT_cmb_dt

    def C_c(self, r_i):
        return 4*pi*r_i**2*p.delta_rho_c/(self.compute_mass_of_partial_core(p.r_c, r_i)*p.alpha_c)

    def compute_Lhp(self, r_i, dP = 1.):
        # P_icb = self.P(r_i)
        # dTm_dP = (self.T_m(P_icb)-self.T_m(P_icb+dP))/dP
        return p.L_H

    def Q_L(self, dT_cmb_dt, r_i):
        dr_i_dt = self.dr_i_dt(dT_cmb_dt)
        return 4*pi*r_i**2*self.compute_Lhp(r_i)*self.rho(r_i)*dr_i_dt

    def Qt_L(self, r_i):
        return 4*pi*r_i**2*self.compute_Lhp(r_i)*self.rho(r_i)*p.C_r

    def E_L(self, T_cmb, dT_cmb_dt, r_i, Q_L=None):
        dr_i_dt = self.dr_i_dt(dT_cmb_dt)
        if Q_L is None:
            Q_L = self.Q_L(r_i, dr_i_dt)
        T_i = self.T_adiabat_from_T_cmb(T_cmb, r_i)
        return Q_L*(T_i-T_cmb)/(T_i*T_cmb)

    def Et_L(self, T_cmb, r_i, Qt_L=None):
        if Qt_L is None:
            Qt_L = self.Qt_L(r_i)
        T_i = self.T_adiabat_from_T_cmb(T_cmb, r_i)
        return Qt_L*(T_i-T_cmb)/(T_i*T_cmb)

    def phi(self, r):
        2/3*pi*p.G*p.rho_cen*r**2*(1-3*r**2/(10*p.L**2)) - 2/3*pi*p.G*p.rho_cen*p.r_c**2*(1-3*p.r_c**2/(10*p.L**2))

    def Q_g(self, dT_cmb_dt, r_i, I_g=None):
        if I_g is None:
            I_g = self.I_g(r_i)
        return I_g*p.alpha_c*self.C_c(r_i)*p.C_r*dT_cmb_dt

    def Qt_g(self, r_i, I_g=None):
        if I_g is None:
            I_g = self.I_g(r_i)
        return I_g*p.alpha_c*self.C_c(r_i)*p.C_r


    def I_g(self, r_i):
        Csq = 3*p.L**2/16 - p.r_c**2/2*(1-3*p.r_c**2/(10*p.L**2))
        return 8*pi**2*p.rho_cen**2*p.G/3*(
            (3/20*p.r_c**5 - p.L**2/8*p.r_c**3 - p.L**2*Csq*p.r_c)*exp(-p.r_c**2/p.L**2) + Csq/2*p.L**3*pi**0.5*spec.erf(p.r_c/p.L)
            -(3/20*r_i**5 - p.L**2/8*r_i**3 - p.L**2*Csq*r_i)*exp(-r_i**2/p.L**2) + Csq/2*p.L**3*pi**0.5*spec.erf(r_i/p.L)
        )

    def E_g(self, T_cmb, Q_g=None, dr_i_dt=None, r_i=None):
        if Q_g is None:
            Q_g = self.Q_g(dr_i_dt, r_i)
        return Q_g/T_cmb

    def Et_g(self, T_cmb, Qt_g=None, r_i=None):
        if Qt_g is None:
            Qt_g = self.Qt_g(r_i)
        return Qt_g/T_cmb

    def E_k(self):
        return 16*pi*p.k*p.r_c**5/(5*p.D**4)*(1+2/(7*p.D_k**2/p.r_c**2 -1))

    def Q_k(self, T_cmb):
        return 8*pi*p.r_c**3*p.k*T_cmb/p.D**2

    def Qt_T(self, T_cmb, r_i):
        I_s = self.I_s(T_cmb)
        I_g = self.I_g(r_i)
        return self.Qt_g(r_i, I_g=I_g) + self.Qt_L(r_i) + self.Qt_s(T_cmb, I_s=I_s)

    def Et_T(self, T_cmb, r_i):
        return self.Et_g(T_cmb, r_i) + self.Et_s(T_cmb) + self.Et_L(T_cmb, r_i)

    def Q_cmb(self, T_cmb, dT_cmb_dt, r_i, h):
        return self.Q_R(h) + self.Qt_T(T_cmb, r_i)*dT_cmb_dt

    def Delta_E(self, T_cmb, dT_cmb_dt, r_i, h):
        return self.E_R(h, T_cmb) + self.Et_T(T_cmb, r_i)*dT_cmb_dt - self.E_k()

    def T_R(self, T_cmb, h):
        return self.Q_R(h)/self.E_R(h, T_cmb)

    def inner_core_radius(self, T_cmb):
        TaTm = lambda r: self.T_adiabat_from_T_cmb(T_cmb, r)-self.T_m(self.P(r))
        if T_cmb < self.T_m(self.P(p.r_c)):
            return p.r_c
        elif self.T_cen_from_T_cmb(T_cmb) > self.T_m(self.P(0.)):
            return 0.
        else:
            return opt.brentq(TaTm, p.P(p.r_c), p.P(0.))

    def E_phi(self, T_cmb, dT_cmb_dt, r_i, h, T_R):
        Et_T = self.Et_T(T_cmb, r_i)
        Qt_T = self.Qt_T(T_cmb, r_i)
        Q_cmb = self.Q_cmb(T_cmb, dT_cmb_dt, r_i, h)
        return (Q_cmb - self.Q_R(h)*(1-Qt_T/Et_T/T_R)) *Et_T/Qt_T - self.E_k()