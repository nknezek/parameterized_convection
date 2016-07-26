import input_parameters as params
import numpy as np
import scipy.special as spec
from numpy import pi, exp
import scipy.optimize as opt
import scipy.integrate as integrate

class Nimmo_2015(params.Parameters):
    def __init__(self):
        source = 'Nimmo 2015'
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
        # self.k = 120 # W/m-K
        self.D = 6203e3 # m
        self.D_k = 5900e3 # pg. 42
        self.C_p = 840 # J/kg-K
        self.alpha_c = 1.1 # -
        self.delta_rho_c = 560 # kg/m^3
        self.C_r = -10100 # m/K
        self.G = 6.67408e-11 # m^3/kg-s
        self.h_0 = 1.e-11 # - [W/kg] similar to Stevenson Table I
        self.lam = 1.38e-17 # - [1/s] from Stevenson Table I

p = Nimmo_2015()

class core_energetics():
    def __init__(self, c_0, T_cmb0):
        self.c = c_0
        self.T_i = 0.
        self.T_cmb = T_cmb0
        self.compute_mass_of_core()
        self.reset_current_values()
    
    def reset_current_values(self):
        self.current_values = params.Parameters('self')
        self.current_values.C_r = None
        self.current_values.C_c = None
        self.current_values.I_s = None
        self.current_values.I_T = None
        self.current_values.I_g = None
        self.current_values.dr_i_dt = None
        self.current_values.Dc_Dt = None
        self.current_values.T_R = None
        self.current_values.r_i = None
        self.current_values.Q_s = None
        self.current_values.Qt_s = None
        self.current_values.E_s = None
        self.current_values.Et_s = None
        self.current_values.Q_R = None
        self.current_values.E_R = None
        self.current_values.Q_L = None
        self.current_values.Qt_L = None
        self.current_values.E_L = None
        self.current_values.Et_L = None
        self.current_values.Qt_g = None
        self.current_values.Q_g = None
        self.current_values.Et_g = None
        self.current_values.E_g = None
        self.current_values.E_k = None
        self.current_values.Q_k = None
        self.current_values.Qt_T = None
        self.current_values.Et_T = None
        self.current_values.Q_cmb = None
        self.current_values.Delta_E = None
        self.current_values.E_phi = None
        self.current_values.Q_phi = None

    def rho(self, r):
        return p.rho_cen*exp(-r**2/p.L**2)

    def g(self, r):
        return 4*pi/3*p.G*p.rho_cen * r * (1- 3*r**2/(5*p.L**2))

    def P(self, r):
        return p.P_c+4*pi*p.G*p.rho_cen**2/3 * ((3*p.r_c**2/10 - p.L**2/5)*exp(-p.r_c**2/p.L**2) - (3*r**2/10 - p.L**2/5)*exp(-r**2/p.L**2))

    def T_m(self, P):
        return p.T_m0*(1 + p.T_m1*P + p.T_m2*P**2)

    def T_adiabat_from_T_cen(self, T_cen, r):
        return T_cen*exp(-r**2/p.D**2)

    def T_adiabat_from_T_cmb(self, T_cmb, r):
        T_cen = self.T_cen_from_T_cmb(T_cmb)
        return T_cen*exp(-r**2/p.D**2)

    def T_cen_from_T_cmb(self, T_cmb):
        return T_cmb*exp(p.r_c**2/p.D**2)

    def compute_mass_of_core(self):
        self.mass = self.compute_mass_of_partial_core(p.r_c, 0.)
        return self.mass

    def compute_mass_of_partial_core(self, r_top, r_bottom):
        return 4*pi*p.rho_cen*((-p.L**2/2*r_top * exp(-r_top**2/p.L**2) + p.L**3/4*pi**0.5*spec.erf(r_top/p.L))
                           -(-p.L**2/2*r_bottom * exp(-r_bottom**2/p.L**2) + p.L**3/4*pi**0.5*spec.erf(r_bottom/p.L)))

    def C_r(self, T_cmb, r_i=None, recompute=False, store_computed=True):
        if self.current_values.C_r is not None and not recompute:
            return self.current_values.C_r
        else:
            dT = 1e-6
            r_i = self.r_i(T_cmb, recompute=True, store_computed=False)
            r_ip = self.r_i(T_cmb+dT, recompute=True, store_computed=False)
            C_r = (r_ip-r_i)/dT
            if store_computed:
                self.current_values.C_r = C_r
            return C_r
        # def C_r(self, T_cmb, r_i=None):
        #     '''
        #     from Nimmo 2015, eq. [49]
        #     :param T_cmb:
        #     :param r_i:
        #     :return:
        #     '''
        #     dr = 1e-3
        #     if r_i is None:
        #         r_i = self.r_i(T_cmb)
        #     T_i = self.T_adiabat_from_T_cmb(T_cmb, r_i)
        #     rho_i = self.rho(r_i)
        #     P_icb = self.P(r_i)
        #     g_i = self.g(r_i)
        #     P_icbp = self.P(r_i-dr)
        #     # print(r_i, dr)
        #     # print(P_icb, P_icbp)
        #     dTm_dP = (self.T_m(P_icbp) - self.T_m(P_icb))/(P_icbp-P_icb)
        #     dTa_dP = (self.T_adiabat_from_T_cmb(T_cmb, r_i-dr) - self.T_adiabat_from_T_cmb(T_cmb, r_i))/(P_icbp-P_icb)
        #     # print(dTm_dP, dTa_dP)
        #     C_r = -1/(dTm_dP-dTa_dP)*T_i/(rho_i*g_i*T_cmb)
        #     return C_r

    def C_c(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.C_c is not None and not recompute:
            return self.current_values.C_c
        else:
            C_c = 4*pi*self.r_i(T_cmb)**2*p.delta_rho_c/(self.compute_mass_of_partial_core(p.r_c, self.r_i(T_cmb))*p.alpha_c)
            if store_computed:
                self.current_values.C_c = C_c
            return C_c

    def I_s(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.I_s is not None and not recompute:
            return self.current_values.I_s
        else:
            T_cen = self.T_cen_from_T_cmb(T_cmb)
            A = (1/p.L**2 + 1/p.D**2)**-0.5
            I_s = 4*pi*T_cen*p.rho_cen*(-A**2*p.r_c/2*exp(-p.r_c**2/A**2) + A**3*pi**0.5/4*spec.erf(p.r_c/A))
            if store_computed:
                self.current_values.I_s = I_s
            return I_s
        # def I_s_exp(self, T_cmb, recompute=False, store_computed=True):
        #     if self.current_values.I_s_exp is not None and not recompute:
        #         return self.current_values.I_s_exp
        #     else:
        #         T_cen = self.T_cen_from_T_cmb(T_cmb)
        #         A = (1/p.L**2 + 1/p.D**2)**-0.5
        #         I_s_exp = T_cen *4/3 * pi * p.rho_cen * p.r_c**3 * exp(-p.r_c**2/A**2) * (1 + 2*p.r_c**2/(5*A**2))
        #         if store_computed:
        #             self.current_values.I_s_exp = I_s_exp
        #         return I_s_exp

    def I_T(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.I_T is not None and not recompute:
            return self.current_values.I_T
        else:
            T_cen = self.T_cen_from_T_cmb(T_cmb)
            Bsq = (1/p.L**2 - 1/p.D**2)**-1
            I_T = 4*pi*p.rho_cen/(3*T_cen) * p.r_c**3 * (1-3*p.r_c**2/(5*Bsq))
            if store_computed:
                self.current_values.I_T = I_T
            return I_T

    def I_g(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.I_g is not None and not recompute:
            return self.current_values.I_g
        else:
            r_i = self.r_i(T_cmb)
            Csq = 3*p.L**2/16 - p.r_c**2/2*(1 - 3*p.r_c**2/(10*p.L**2))
            I_g = 8*pi**2*p.rho_cen**2*p.G/3*(
                (3/20*p.r_c**5 - p.L**2/8*p.r_c**3 - p.L**2*Csq*p.r_c)*exp(-p.r_c**2/p.L**2) + Csq/2*p.L**3*pi**0.5*spec.erf(p.r_c/p.L)
                -((3/20*r_i**5 - p.L**2/8*r_i**3 - p.L**2*Csq*r_i)*exp(-r_i**2/p.L**2) + Csq/2*p.L**3*pi**0.5*spec.erf(r_i/p.L))
            )
            if store_computed:
                self.current_values.I_g = I_g
            return I_g

    def phi(self, r):
        return (2/3*pi*p.G*p.rho_cen*r**2*(1-3*r**2/(10*p.L**2))
                - (2/3*pi*p.G*p.rho_cen*p.r_c**2*(1-3*p.r_c**2/(10*p.L**2))))

    def dr_i_dt(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.dr_i_dt is not None and not recompute:
            return self.current_values.dr_i_dt
        else:
            dr_i_dt = self.C_r(T_cmb)*dT_cmb_dt
            if store_computed:
                self.current_values.dr_i_dt = dr_i_dt
            return dr_i_dt

    def Dc_Dt(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.Dc_Dt is not None and not recompute:
            return self.current_values.Dc_Dt
        else:
            Dc_Dt = self.C_c(T_cmb)*self.C_r(T_cmb)*dT_cmb_dt
            if store_computed:
                self.current_values.Dc_Dt = Dc_Dt
            return

    def compute_Lhp(self, T_cmb, dP = 1.):
        # P_icb = self.P(r_i)
        # dTm_dP = (self.T_m(P_icb)-self.T_m(P_icb+dP))/dP
        return p.L_H

    def heat_production_per_kg(self, time):
        '''
        Equation (2) from Stevenson et al 1983
        '''
        return p.h_0*np.exp(-p.lam*time)

    def T_R(self, T_cmb, h, recompute=False, store_computed=True):
        if self.current_values.T_R is not None and not recompute:
            return self.current_values.T_R
        else:
            if h == 0.:
                T_R = 1e99
            else:
                T_R = self.Q_R(h)/self.E_R(T_cmb, h)
            if store_computed:
                self.current_values.T_R = T_R
            return T_R

    def r_i(self, T_cmb, recompute=False, store_computed=True, one_off=False):
        if self.current_values.r_i is not None and not recompute and not one_off:
            return self.current_values.r_i
        else:
            TaTm = lambda r: self.T_adiabat_from_T_cmb(T_cmb, r)-self.T_m(self.P(r))
            if T_cmb < self.T_m(self.P(p.r_c)):
                r_i = p.r_c
            elif self.T_cen_from_T_cmb(T_cmb) > self.T_m(self.P(0.)):
                r_i = 0.
            else:
                r_i = opt.brentq(TaTm, p.r_c, 0.)
            if store_computed:
                self.current_values.r_i = r_i
            return r_i

    def Qt_s(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Qt_s is not None and not recompute:
            return self.current_values.Qt_s
        else:
            Qt_s = -p.C_p/T_cmb*self.I_s(T_cmb)
            if store_computed:
                self.current_values.Qt_s = Qt_s
            return Qt_s

    def Q_s(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.Q_s is not None and not recompute:
            return self.current_values.Q_s
        else:
            Q_s =  -p.C_p/T_cmb*dT_cmb_dt*self.I_s(T_cmb)
            if store_computed:
                self.current_values.Q_s = Q_s
            return Q_s

    def Et_s(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Et_s is not None and not recompute:
            return self.current_values.Et_s
        else:
            Et_s = p.C_p/T_cmb * (self.mass - self.I_s(T_cmb)/T_cmb)
            if store_computed:
                self.current_values.Et_s = Et_s
            return Et_s

    def E_s(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.E_s is not None and not recompute:
            return self.current_values.E_s
        else:
            E_s = p.C_p/T_cmb * (self.mass-self.I_s(T_cmb)/T_cmb)*dT_cmb_dt
            if store_computed:
                self.current_values.E_s = E_s
            return E_s

    def Q_R(self, h, recompute=False, store_computed=True):
        if self.current_values.Q_R is not None and not recompute:
            return self.current_values.Q_R
        else:
            Q_R = self.mass*h
            if store_computed:
                self.current_values.Q_R = Q_R
            return Q_R

    def E_R(self, T_cmb, h, recompute=False, store_computed=True):
        if self.current_values.E_R is not None and not recompute:
            return self.current_values.E_R
        else:
            E_R = (self.mass/T_cmb - self.I_T(T_cmb))*h
            if store_computed:
                self.current_values.E_R = E_R
            return E_R

    def Qt_L(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Qt_L is not None and not recompute:
            return self.current_values.Qt_L
        else:
            r_i = self.r_i(T_cmb)
            Qt_L = 4*pi*r_i**2*self.compute_Lhp(T_cmb)*self.rho(r_i)*self.C_r(T_cmb)
            if store_computed:
                self.current_values.Qt_L = Qt_L
            return Qt_L

    def Q_L(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.Q_L is not None and not recompute:
            return self.current_values.Q_L
        else:
            Q_L = self.Qt_L(T_cmb)*dT_cmb_dt
            if store_computed:
                self.current_values.Q_L = Q_L
            return Q_L

    def Et_L(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Et_L is not None and not recompute:
            return self.current_values.Et_L
        else:
            T_i = self.T_adiabat_from_T_cmb(T_cmb, self.r_i(T_cmb))
            Et_L = self.Qt_L(T_cmb)*(T_i-T_cmb)/(T_i*T_cmb)
            if store_computed:
                self.current_values.Et_L = Et_L
            return Et_L

    def E_L(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.E_L is not None and not recompute:
            return self.current_values.E_L
        else:
            E_L = self.Et_L(T_cmb)*dT_cmb_dt
            if store_computed:
                self.current_values.E_L = E_L
            return E_L

    def Qt_g(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Qt_g is not None and not recompute:
            return self.current_values.Qt_g
        else:
            M_oc = self.compute_mass_of_partial_core(p.r_c, self.r_i(T_cmb))
            Qt_g = (self.I_g(T_cmb)-M_oc*self.phi(self.r_i(T_cmb)))*p.alpha_c*self.C_c(T_cmb)*self.C_r(T_cmb)
            if store_computed:
                self.current_values.Qt_g = Qt_g
            return Qt_g

    def Q_g(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.Q_g is not None and not recompute:
            return self.current_values.Q_g
        else:
            Q_g = self.Qt_g(T_cmb)*dT_cmb_dt
            if store_computed:
                self.current_values.Q_g = Q_g
            return Q_g

    def Et_g(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Et_g is not None and not recompute:
            return self.current_values.Et_g
        else:
            Et_g = self.Qt_g(T_cmb)/T_cmb
            if store_computed:
                self.current_values.Et_g = Et_g
            return Et_g

    def E_g(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        if self.current_values.E_g is not None and not recompute:
            return self.current_values.E_g
        else:
            E_g = self.Q_g(T_cmb, dT_cmb_dt)/T_cmb
            if store_computed:
                self.current_values.E_g = E_g
            return E_g

    def Q_k(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Q_k is not None and not recompute:
            return self.current_values.Q_k
        else:
            Q_k = 8*pi*p.r_c**3*p.k*T_cmb/p.D**2
            if store_computed:
                self.current_values.Q_k = Q_k
            return Q_k

    def E_k(self, recompute=False, store_computed=True):
        if self.current_values.E_k is not None and not recompute:
            return self.current_values.E_k
        else:
            E_k = 16*pi*p.k*p.r_c**5/(5*p.D**4)
            if store_computed:
                self.current_values.E_k = E_k
            return E_k
        # def E_k(self):
        #     return 16*pi*p.k*p.r_c**5/(5*p.D**4)*(1+2/(7*p.D_k**2/p.r_c**2-1))

    def Qt_T(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Qt_T is not None and not recompute:
            return self.current_values.Qt_T
        else:
            Qt_T = self.Qt_g(T_cmb) + self.Qt_L(T_cmb) + self.Qt_s(T_cmb)
            if store_computed:
                self.current_values.Qt_T = Qt_T
            return Qt_T

    def Et_T(self, T_cmb, recompute=False, store_computed=True):
        if self.current_values.Et_T is not None and not recompute:
            return self.current_values.Et_T
        else:
            Et_T = self.Et_g(T_cmb) + self.Et_L(T_cmb) + self.Et_s(T_cmb)
            if store_computed:
                self.current_values.Et_T = Et_T
            return Et_T

    def Q_cmb(self, T_cmb, dT_cmb_dt, h, recompute=False, store_computed=True):
        if self.current_values.Q_cmb is not None and not recompute:
            return self.current_values.Q_cmb
        else:
            Q_cmb = self.Q_R(h) + self.Qt_T(T_cmb)*dT_cmb_dt
            if store_computed:
                self.current_values.Q_cmb = Q_cmb
            return Q_cmb

    def Delta_E(self, T_cmb, dT_cmb_dt, h, recompute=False, store_computed=True):
        if self.current_values.Delta_E is not None and not recompute:
            return self.current_values.Delta_E
        else:
            Delta_E = self.E_R(T_cmb, h) + self.Et_T(T_cmb)*dT_cmb_dt - self.E_k()
            if store_computed:
                self.current_values.Delta_E = Delta_E
            return Delta_E

    def Q_phi(self, T_cmb, dT_cmb_dt, h, T_D, recompute=False, store_computed=True):
        if self.current_values.E_phi is not None and not recompute:
            return self.current_values.E_phi
        else:
            Q_phi = self.E_phi(T_cmb, dT_cmb_dt, h)*T_D
            if store_computed:
                self.current_values.Q_phi = Q_phi
            return Q_phi

    def E_phi(self, T_cmb, dT_cmb_dt, h, recompute=False, store_computed=True):
        if self.current_values.E_phi is not None and not recompute:
            return self.current_values.E_phi
        else:
            Et_T = self.Et_T(T_cmb)
            Qt_T = self.Qt_T(T_cmb)
            T_R = self.T_R(T_cmb, h)
            Q_cmb = self.Q_cmb(T_cmb, dT_cmb_dt, h)
            E_phi = (Q_cmb - self.Q_R(h)*(1-Qt_T/Et_T/T_R)) *Et_T/Qt_T - self.E_k()
            if store_computed:
                self.current_values.E_phi = E_phi
            return E_phi

    def core_energy_balance(self, time, T_cmb, Q_cmb):
        Qt_T = self.Qt_T(T_cmb)
        Q_R = self.Q_R(self.heat_production_per_kg(time))
        dT_cmb_dt = (Q_cmb - Q_R)/Qt_T
        return dT_cmb_dt