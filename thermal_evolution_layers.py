import numpy as np
import scipy.integrate as integrate
import scipy.special as spec
from numpy import pi, exp
import scipy.optimize as opt
import solubility_library as sol
import scipy.special as sp

const_yr_to_sec = 3.154e7 # seconds

class Parameters(object):
    def __init__(self, source):
        self.source = source

class Planet(object):
    def __init__(self, layers, params):
        self.layers = layers
        self.Nlayers = len(layers)

        self.radius = self.layers[-1].outer_radius
        self.volume = 4. / 3. * np.pi * self.radius ** 3
        for layer in self.layers:
            layer.planet = self
        self.params = params

    def integrate(self, x0, xkey=None):
        raise NotImplementedError('must implement an integrate method')

class Layer(object):
    '''
    The layer base class defines the geometry of a spherical shell within
    a planet.
    '''

    def __init__(self, inner_radius, outer_radius, params={}):
        self.set_boundaries(inner_radius, outer_radius)
        self.params = params

    def set_boundaries(self, inner_radius, outer_radius):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.thickness = outer_radius - inner_radius

        assert self.thickness >= 0.0, "Ri={0:.1f} km, Ro={1:.1f} km".format(self.inner_radius / 1e3,
                                                                            self.outer_radius / 1e3)

        self.inner_surface_area = 4.0 * np.pi * self.inner_radius ** 2.
        self.outer_surface_area = 4.0 * np.pi * self.outer_radius ** 2.

        self.volume = 4.0 / 3.0 * np.pi * (self.outer_radius ** 3. - self.inner_radius ** 3.)

class CoreLayer(Layer):
    def __init__(self, inner_radius=0., outer_radius=3480e3, params={}):
        Layer.__init__(self, inner_radius, outer_radius, params)

class MantleLayer(Layer):
    def __init__(self, inner_radius, outer_radius, params=None):
        Layer.__init__(self, inner_radius, outer_radius, params)

class Planet_Stevenson(Planet):
    def __init__(self, case=1):
        params = Parameters("Stevenson 1983")
        Planet.__init__(self, [CoreLayer_Stevenson(params=params, case=case), MantleLayer_Stevenson(params=params)], params)
        self.core_layer = self.layers[0]
        self.mantle_layer = self.layers[1]

    def ODE(self, x, t):
        T_cmb = x[0]
        T_um = x[1]
        try:
            self.params.mantle.verbose
            vm = True
        except:
            vm = False
        dTm_dt = self.mantle_layer.energy_balance(T_cmb, T_um, t, verbose=vm)
        cmb_flux = self.mantle_layer.lower_boundary_flux(T_cmb, T_um)
        dTc_dt = self.core_layer.energy_balance(T_cmb, cmb_flux)
        return np.array([dTc_dt, dTm_dt])

    def integrate(self, times, x0, full_output=False):
        solution = integrate.odeint(self.ODE, x0, times, full_output=full_output)
        return solution

class CoreLayer_Stevenson(CoreLayer):
    def __init__(self, params=None, case=1):
        if params is None:
            params = Parameters('Stevenson 1983 for core')
        self.params = params
        self.params.core = Parameters("Stevenson 1983 for core")
        pc = self.params.core
        pc.R_c0 = 3485e3
        pc.rho = 13000. # - [kg/m^3] from Stevenson 1983 pg. 474
        pc.g = 10. # - [m/s^2] from Stevenson 1983 Table II
        pc.alpha = 2e-5 # - [/K] from Stevenson 1983 Table I
        pc.rhoC = 4e6 # - [J/m^3-K] from Stevenson 1983 Table I
        pc.C = pc.rhoC/pc.rho
        pc.x_0 = 0.1 # - [wt% S] from Stevenson 1983 pg. 474
        pc.P_c = 360e9 # - [Pa] from Stevenson 1983 pg. 474
        pc.P_cm = 140e9 # - [Pa] from Stevenson 1983 pg. 474
        pc.mu = 1.2 # - [] from Stevenson 1983 pg. 473 and Table II
        pc.T_m1 = 6.14e-12 # - [K/Pa] from Stevenson 1983 Table II
        pc.T_m2 = -4.5e-24 # - [K/Pa^2] from Stevenson 1983 Table II
        pc.T_a1 = 3.96e-12 # - [K/Pa] from Stevenson 1983 Table II
        pc.T_a2 = -3.3e-24 # - [K/Pa^2] from Stevenson 1983 Table II
        self.set_inner_core_L_Eg(case, params)
        Layer.__init__(self, 0., pc.R_c0, params)
        self.light_alloy = pc.x_0

    def set_inner_core_L_Eg(self, case, params):
        if case == 1:
            params.core.L_Eg = 1e6  # - [J/kg] from self Table III
            params.core.T_m0 = 1950.  # - [K] from self Table III
        elif case == 2:
            params.core.L_Eg = 2e6  # - [J/kg] from self Table III
            params.core.T_m0 = 1980.  # - [K] from self Table III
        else:
            raise ValueError("case must be integer 1 for E1 or 2 for E2")

    def set_light_alloy_concentration(self):
        '''
        Equation (7) from Stevenson 1983
        '''
        pc = self.params.core
        R_c = self.inner_radius
        R_i = self.outer_radius
        self.light_alloy = pc.x_0 * (R_c ** 3) / (R_c ** 3 - R_i ** 3)
        return self.light_alloy

    def set_inner_core_radius(self, R_i):
        self.inner_radius = R_i
        return self.inner_radius

    def T_cmb(self):
        return self.T_average / self.mu

    def T_m(self, P):
        '''
        Equation (3) from Stevenson 1983

        Calculates the liquidus temp for a given pressure in the core P
        '''
        x = self.light_alloy
        pc = self.params.core
        return pc.T_m0 * (1. - pc.alpha * x) * (1. + pc.T_m1 * P + pc.T_m2 * P ** 2.)

    def T_a(self, P, T_cmb):
        '''
        Equation (4) from Stevenson 1983

        Calculates adiabatic temperature for a given pressure within the core P, given the temperature at the CMB T_cmb
        '''
        pc = self.params.core
        return T_cmb * (1. + pc.T_a1 * P + pc.T_a2 * P ** 2.) / (1. + pc.T_a1 * pc.P_cm + pc.T_a2 * pc.P_cm ** 2.)

    def P_io(self, T_cmb):
        pc = self.params.core
        opt_function = lambda P: (self.T_a(P, T_cmb) - self.T_m(P))
        if self.T_m(pc.P_c) <= self.T_a(pc.P_c, T_cmb):
            P_io = pc.P_c
        elif self.T_m(pc.P_cm) >= self.T_a(pc.P_cm, T_cmb):
            P_io = pc.P_cm
        else:
            C = pc.T_m0*(1. + pc.T_a1*pc.P_cm + pc.T_a2*pc.P_cm**2.)*(1-pc.alpha*self.light_alloy)
            Cma1 = (C*pc.T_m1-pc.T_a1*T_cmb)
            Cma2 = (C*pc.T_m2-pc.T_a2*T_cmb)
            sqr = (Cma1**2 - 4*(C-T_cmb)*Cma2)**0.5
            P_io = (-Cma1 + sqr)/(2*Cma2)
        return P_io

    def P_io_old(self, T_cmb):
        pc = self.params.core
        opt_function = lambda P: (self.T_a(P, T_cmb) - self.T_m(P))
        if self.T_m(pc.P_c) <= self.T_a(pc.P_c, T_cmb):
            P_io = pc.P_c
        elif self.T_m(pc.P_cm) >= self.T_a(pc.P_cm, T_cmb):
            P_io = pc.P_cm
        else:
            P_io = opt.brentq(opt_function, pc.P_c, pc.P_cm)
        return P_io

    def dPio_dTcmb(self, T_cmb):
        '''
        Calculate the derivative of ICB pressure with Tcmb using simultaneous solution of eq (3) and (4) of Stevenson 1983
        '''
        pc = self.params.core
        C = pc.T_m0*(1. + pc.T_a1 * pc.P_cm + pc.T_a2 * pc.P_cm ** 2.) * (1 - pc.alpha * self.light_alloy)
        Cma1 = (C * pc.T_m1 - pc.T_a1 * T_cmb)
        Cma2 = (C * pc.T_m2 - pc.T_a2 * T_cmb)
        sqr = (Cma1**2-4*(C-T_cmb)*Cma2)**0.5
        dPio_dTcmb = (pc.T_a1/(2*Cma2) + (-pc.T_a1*Cma1 + 2*Cma2 + 2*pc.T_a2*(C-T_cmb))/(2*Cma2*sqr)
                        + pc.T_a2/(2*Cma2)*(sqr + pc.T_a1*T_cmb-C*pc.T_m1))
        return dPio_dTcmb

    def R_i(self, T_cmb):
        '''
        Equation 5 from Stevenson et al 1983
        '''
        pc = self.params.core
        R_c = self.outer_radius
        P_io = self.P_io(T_cmb)
        R_i = max(0., np.sqrt(2. * (pc.P_c - P_io) * R_c / (pc.rho * pc.g)))
        return R_i

    def dRi_dTcm(self, T_cmb):
        '''
        calculate derivative of ICB radius with Tcmb using simulatneous solution of eq (3-5) of Stevenson 1983
        '''
        pc = self.params.core
        R_c = self.outer_radius
        P_io = self.P_io(T_cmb)
        dPio_dTcmb = self.dPio_dTcmb(T_cmb)
        R_i = self.R_i(T_cmb)
        if R_i > 0.:
            dRi_dTcmb = -R_c/(pc.rho*pc.g*R_i)*dPio_dTcmb
        else:
            dRi_dTcmb = 0.
        return dRi_dTcmb

    def energy_balance(self, T_cmb, core_flux):
        pc = self.params.core
        core_surface_area = self.outer_surface_area
        inner_core_surface_area = np.power(self.R_i(T_cmb), 2.0) * 4. * np.pi
        dRi_dTcmb = 0.
        try:
            # dRi_dTcmb = derivative(self.R_i, T_cmb, dx=1.0)
            dRi_dTcmb = self.dRi_dTcm(T_cmb)
        except ValueError:
            pass
        effective_heat_capacity = pc.rho * pc.C * self.volume * pc.mu
        latent_heat = -pc.L_Eg * pc.rho * inner_core_surface_area * dRi_dTcmb
        dTdt = -core_flux * core_surface_area / (effective_heat_capacity - latent_heat)
        return dTdt

class MantleLayer_Stevenson(MantleLayer):
    def __init__(self, params=None):
        if params is None:
            params = Parameters('Stevenson 1983 for mantle')
        self.params = params
        self.params.mantle = Parameters('Stevenson 1983 for mantle')
        pm = self.params.mantle
        pm.R_p0 = 6371e3  # - [m] from Stevenson 1983 Table II
        pm.R_c0 = 3485e3  # - [m] from Stevenson 1983 pg. 474
        pm.T_s = 293.  # - [K] from Stevenson 1983 Table II
        pm.mu = 1.3 # - [] from Stevenson 1983 pg. 473 and Table II
        pm.alpha = 2e-5 # - [/K] from Stevenson 1983 Table I
        pm.k = 4.0 # - [W/m-K] from Stevenson 1983 Table I
        pm.K = 1e-6 # - [m^2/s] from Stevenson 1983 Table I
        pm.rhoC = 4e6 # - [J/m^3-K] from Stevenson 1983 Table I
        pm.rho = 5000. # - [kg/m^3] -- guess as Stevenson 1983 never explicitly states his assumption for rho or C
        pm.C = pm.rhoC/pm.rho # - [J/K-kg]
        pm.Q_0 = 1.7e-7 # - [W/m^3] from Stevenson 1983 Table I
        pm.lam = 1.38e-17 # - [1/s] from Stevenson 1983 Table I
        pm.A = 5.2e4 # - [K] from Stevenson 1983 Table I
        pm.nu_0 = 4.0e3 # - [m^2/s] from Stevenson 1983 Table I
        pm.Ra_crit = 5e2 # - [] from Stevenson 1983 Table I
        pm.beta = 0.3 # - [] from Stevenson 1983 Table I
        pm.g = 10. # - [m/s^2] from Stevenson 1983 Table II
        pm.Ra_boundary_crit = 2e3 # empirical parameparams

        MantleLayer.__init__(self, pm.R_c0, pm.R_p0, params)

    def adiabat_from_bottom(self, T_magma_ocean_top, distance):
        pm = self.params.mantle
        return T_magma_ocean_top * (1.0 - pm.alpha * pm.g * distance / pm.C)

    def average_mantle_temp(self, T_upper_mantle):
        pm = self.params.mantle
        return T_upper_mantle * pm.mu

    def kinematic_viscosity(self, T_upper_mantle):
        pm = self.params.mantle
        return pm.nu_0 * np.exp(pm.A / T_upper_mantle)

    def find_arrenhius_params(self, nu1, T1, nu2, T2):
        '''
        given a set of two viscosities and temperatures, returns the parameters for Arrehnius visocity relation

        nu(T) = nu0*exp(A/T)

        :param nu1:
        :param T1:
        :param nu2:
        :param T2:
        :return: A, nu0
        '''

        A = np.log(nu1 / nu2) / (1 / T1 - 1 / T2)
        nu0 = nu1*np.exp(-A/T1)
        return A, nu0

    def heat_production(self, time):
        '''
        Equation (2) from Stevenson et al 1983
        '''
        pm = self.params.mantle
        return pm.Q_0 * np.exp(-pm.lam * time)

    def lower_mantle_temperature(self, T_upper_mantle):
        '''
        Adiabatic Temperature Increase from the temperature at the base of upper mantle boundary layer to
        the top of the lower boundary layer assuming negligable boundary layer thickness.
        '''
        pm = self.params.mantle
        return T_upper_mantle * (1.0 + pm.alpha * pm.g * self.thickness / pm.C)

    def mantle_rayleigh_number(self, T_mantle_bottom, T_upper_mantle):
        '''
        Equation (19) Stevenson et al 1983
        '''
        pm = self.params.mantle
        nu = self.kinematic_viscosity(T_upper_mantle)
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        upper_boundary_delta_T = T_upper_mantle - pm.T_s
        lower_boundary_delta_T = T_mantle_bottom - T_lower_mantle
        # assert upper_boundary_delta_T > 0.0
        # assert lower_boundary_delta_T > 0.0
        delta_T_effective = upper_boundary_delta_T + lower_boundary_delta_T
        return pm.g * pm.alpha * (delta_T_effective) * np.power(self.thickness, 3.) / (nu * pm.K)

    def boundary_layer_thickness(self, Ra):
        '''
        Equation (18) Stevenson et al 1983
        '''
        pm = self.params.mantle
        return self.thickness * np.power(pm.Ra_crit / Ra, pm.beta)

    def upper_boundary_layer_thickness(self, T_mantle_bottom, T_upper_mantle):
        '''
        Use Equations (18,19) from Stevenson et al 1983
        '''
        Ra = self.mantle_rayleigh_number(T_mantle_bottom, T_upper_mantle)
        return self.boundary_layer_thickness(Ra)

    def lower_boundary_layer_thickness(self, T_mantle_bottom, T_upper_mantle):
        '''
        Equations (20,21) Stevenson et al 1983

        '''
        pm = self.params.mantle
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        delta_T_lower_boundary_layer = T_mantle_bottom - T_lower_mantle
        average_boundary_layer_temp = T_lower_mantle + delta_T_lower_boundary_layer / 2.
        nu_crit = self.kinematic_viscosity(T_upper_mantle)
        # assert delta_T_lower_boundary_layer > 0.0, "dTlbl={3:.1f} K, T_mb={0:.1f} K, T_lm={1:.1f} K, T_um={2:.1f} K".format(
        #     T_mantle_bottom, T_lower_mantle, T_upper_mantle, delta_T_lower_boundary_layer)
        delta_c = np.power(pm.Ra_boundary_crit * nu_crit * pm.K / (pm.g * pm.alpha * delta_T_lower_boundary_layer),
                           0.333)
        Ra_mantle = self.mantle_rayleigh_number(T_mantle_bottom, T_upper_mantle)
        delta_c_normal = self.boundary_layer_thickness(Ra_mantle)
        return np.minimum(delta_c, delta_c_normal)

    def upper_boundary_flux(self, T_mantle_bottom, T_upper_mantle):
        '''
        Equation (17) from Stevenson et al 1983

        :param T_upper_mantle:
        :param T_mantle_bottom:
        :return:
        '''
        pm = self.params.mantle
        delta_T = T_upper_mantle - pm.T_s
        upper_boundary_layer_thickness = self.upper_boundary_layer_thickness(T_mantle_bottom, T_upper_mantle)
        return pm.k * delta_T / upper_boundary_layer_thickness

    def lower_boundary_flux(self, T_mantle_bottom, T_upper_mantle):
        '''
        Equation (17) from Stevenson et al 1983

        :param T_upper_mantle:
        :param T_mantle_bottom:
        :return:
        '''
        pm = self.params.mantle
        delta_T = T_mantle_bottom - self.lower_mantle_temperature(T_upper_mantle)
        lower_boundary_layer_thickness = self.lower_boundary_layer_thickness(T_mantle_bottom, T_upper_mantle)
        return pm.k * delta_T / lower_boundary_layer_thickness

    def energy_balance(self, T_mantle_bottom, T_upper_mantle, time, verbose=False):
        if verbose:
            print(time/3e7/1e6, T_mantle_bottom, T_upper_mantle)
        pm = self.params.mantle
        mantle_surface_area = self.outer_surface_area
        core_surface_area = self.inner_surface_area

        effective_heat_capacity =  pm.rho * pm.C * pm.mu * self.volume
        internal_heat_energy = self.heat_production(time) * self.volume
        cmb_flux = self.lower_boundary_flux(T_mantle_bottom, T_upper_mantle)
        surface_flux = self.upper_boundary_flux(T_mantle_bottom, T_upper_mantle)
        net_flux_out = mantle_surface_area * surface_flux - core_surface_area * cmb_flux
        dTdt = (internal_heat_energy - net_flux_out) / effective_heat_capacity
        return dTdt

class Planet_Stevenson_backward(Planet):
    def __init__(self, case=1):
        params = Parameters("Stevenson 1983")
        Planet.__init__(self, [CoreLayer_Stevenson_backwards(params=params, case=case), MantleLayer_Stevenson_backwards(params=params)], params)
        self.core_layer = self.layers[0]
        self.mantle_layer = self.layers[1]

    def ODE(self, x, t):
        T_cmb = x[0]
        T_um = x[1]
        dTm_dt = self.mantle_layer.energy_balance(T_cmb, T_um, t)
        cmb_flux = self.mantle_layer.lower_boundary_flux(T_cmb, T_um)
        dTc_dt = self.core_layer.energy_balance(T_cmb, cmb_flux)
        return np.array([-dTc_dt, -dTm_dt])

    def integrate(self, times, x0, full_output=False):
        solution = integrate.odeint(self.ODE, x0, times, full_output=full_output)
        return solution

class CoreLayer_Stevenson_backwards(CoreLayer_Stevenson):
    def __init__(self, params=None, case=1):
        CoreLayer_Stevenson.__init__(self, params=params, case=case)

class MantleLayer_Stevenson_backwards(MantleLayer_Stevenson):
    def __init__(self, params=None):
        MantleLayer_Stevenson.__init__(self, params)

    def heat_production(self, time):
        '''
        Equation (2) from Stevenson et al 1983
        '''
        pm = self.params.mantle
        return pm.Q_0 * np.exp(-pm.lam * (pm.time_end-time))

class Planet_NimmoStevenson(Planet):
    def __init__(self, case=1):
        params = Parameters('Stevenson Mantle, Nimmo Core')
        Planet.__init__(self, [CoreLayer_Nimmo(params=params), MantleLayer_Stevenson(params=params)], params)
        self.core_layer = self.layers[0]
        self.mantle_layer = self.layers[1]

    def ODE(self, x, t):
        T_cmb = x[0]
        T_um = x[1]
        try:
            self.params.mantle.verbose
            vm = True
        except:
            vm = False
        dTm_dt = self.mantle_layer.energy_balance(T_cmb, T_um, t, verbose=vm)
        cmb_flux = self.mantle_layer.lower_boundary_flux(T_cmb, T_um)
        dTc_dt = self.core_layer.energy_balance(t, T_cmb, cmb_flux)
        return np.array([dTc_dt, dTm_dt])

    def integrate(self, times, x0, full_output=False):
        solution = integrate.odeint(self.ODE, x0, times, full_output=full_output)
        return solution

class CoreLayer_Labrosse(CoreLayer):
    def __init__(self, params=None, T_cmb0 = None):
        if params is None:
            params = Parameters('Labrosse 2015 Core')
        self.params = params
        self.params.core = Parameters('Labrosse 2015 Core')
        pc = self.params.core
        pc.rho_0 = 12451 # [kg/m^3] from Labrosse Table 1
        pc.L_rho = 8039e3 # [m] from Labrosse Table 1
        pc.A_rho = 0.484 # [-] from Labrosse Table 1
        pc.K_0 = 1403e9  # [Pa] from Labrosse Table 1
        pc.K_0p = 3.567# [-] from Labrosse Table 1
        pc.r_oc = 3480e3 # [m] from Labrosse Table 2
        pc.r_icp = 1221e3 # [m] from Labrosse Table 2
        pc.k_0 = 163 # [W/m-K] from Labrosse Table 2
        pc.A_k = 2.39 # [-] from Labrosse Table 2
        pc.gruneisen = 1.5 # [-] from Labrosse Table 2
        pc.C_p = 750 # [J/K-kg] from Labrosse Table 2
        pc.DS = 127 # [J/K-kg] from Labrosse Table 2
        pc.beta = 0.83 # [-] from Labrosse Table 2
        pc.dTldX = -21e3 # [K] from Labrosse Table 2
        pc.dTldP = 9e-9 # [K/Pa] from Labrosse Table 2
        pc.DX_icb = 0.056 # [-] from Labrosse Table 2
        pc.T_l_r_icp = 5500 # [K] from Labrosse Table 2
        pc.DX_rho = 580 # [kg/m^3] from Labrosse Table 2
        pc.G = 6.67408e-11 # [m^3/kg-s^2] Gravitational Constant
        pc.P0 = 350e9 # [Pa] guess, no listed in Labrosse 2015
        pc.T_cmb0 = T_cmb0 # [K] initial cmb temperature
        pc.M_c = self.M_oc(0)
        pc.Tc2cmb = (1-(pc.r_oc/pc.L_rho)**2-pc.A_rho*(pc.r_oc/pc.L_rho)**4)**-pc.gruneisen
        CoreLayer.__init__(self, outer_radius=params.core.r_oc, params=params)

    def rho_a(self, r):
        '''
        adiabatic density from eq (5) Labrosse 2015

        :param r: radius [m]
        :return: rho_a [kg/m^3]
        '''
        pc = self.params.core
        rho_a = pc.rho_0*(1-(r/pc.L_rho)**2 - pc.A_rho*(r/pc.L_rho)**4)
        return rho_a

    def g_a(self, r):
        '''
        adiabatic gravity from eq (6) Labrosse 2015

        :param r:
        :return:
        '''
        pc = self.params.core
        g_a = 4*np.pi/3*pc.G*pc.rho_0*r*(1-3/5*(r/pc.L_rho)**2 - 3*pc.A_rho/7*(r/pc.L_rho)**4)
        return g_a

    def P(self, r):
        '''
        adiabatic pressure eq (7) Labrosse 2015

        :param r:
        :return:
        '''
        pc = self.params.core
        P = pc.P_0 - pc.K_0*((r/pc.L_rho)**2 + 4/5*(r/pc.L_rho)**4)
        return P

    def dg(self, r):
        '''
        eq (9) in Labross 2015, not sure if used

        :param r:
        :return:
        '''
        # TODO
        # dg = 4*np.pi/3*pc.G*
        pass

    def k(self, r):
        '''
        Thermal Conductivity from eq (20) in Labrosse 2015

        :param r:
        :return:
        '''
        pc = self.params.core
        k = pc.k_0*(1-pc.A_k*r**2/pc.L_rho**2)
        return k

    def T_l(self, r):
        pc = self.params.core
        xi_0 = None #TODO
        T_l0 = self.T_a_from_T_cmb(0, T_cmb0)
        T_l = (T_l0 - pc.K_0*pc.dTldP*(r/pc.L_rho)**2
               + pc.dTldX*xi_0*r**3/(pc.L_rho*self.f_C(pc.r_oc/pc.L_rho,0)))
        return T_l

    def T_a_from_T_c(self, r, T_c):
        '''
        Adiabatic Temperature gradient from Labross eq (15)

        :param rho_a:
        :return:
        '''
        pc = self.params.core
        T_a = T_c*(1-(r/pc.L_rho)**2 - pc.A_rho*(r/pc.L_rho)**4)**pc.gruneisen
        return T_a

    def T_a_from_T_cmb(self, r, T_cmb):
        pc = self.params.core
        T_a = T_cmb*pc.Tc2cmb*(1-(r/pc.L_rho)**2 - pc.A_rho*(r/pc.L_rho)**4)**pc.gruneisen
        return T_a

    def f_C(self, x, delta):
        '''
        integral of density profile, eq (A.2) in Labross 2015

        :param x: quantity to integrate
        :param delta: quantity to integrate
        :return:
        '''
        pc = self.params.core
        return x**3*(1-3/5*(delta+1)*x**2 - 3/14*(delta+1)*(2*pc.A_rho-delta)*x**4)

    def M_oc(self, r_ic):
        '''
        Mass of outer core [kg] given radius of inner core

        :param r_ic: inner-core radius [m]
        :return:
        '''
        pc = self.params.core
        M_oc = 4*np.pi/3*pc.rho_0*pc.L_rho**3*(
            self.f_C(pc.r_oc/pc.L_rho, 0)-self.f_C(r_ic/pc.L_rho,0))
        return M_oc

    def xi(self, t):
        pc = self.params.core
        xi_0 = None #TODO
        r_ic = self.r_ic()
        xi = xi_0*pc.M_c/self.M_oc(r_ic)
        return xi

    def r_ic(self, T_cmb):
        pc = self.params.core
        opt_function = lambda r: self.T_l(r)-self.T_a_from_T_cmb(r,T_cmb)
        r_ic = opt.brentq(opt_function, 0, pc.r_oc)
        return r_ic

class CoreLayer_Nimmo(CoreLayer):
    def __init__(self, params=None):
        if params is None:
            params = Parameters('Parent to Nimmo')
        self.params = params
        self.params.core = Parameters('Nimmo 2015')
        pc = self.params.core
        pc.rho_cen = 12500 # [kg / m^3] from Nimmo 2015 Table 2
        pc.rho_0 = 7900 # [kg/m^3] from Nimmo 2015 Table 2
        pc.r_c = 3480e3 # [m] from Nimmo 2015 Table 2
        pc.r_i = 1220e3 # [m] from Nimmo 2015 Table 2
        pc.K_0 = 500e9 # [Pa] from Nimmo 2015 Table 2
        pc.L = 7272e3 # [m] from Nimmo 2015 Table 2
        pc.P_c = 139e9 # [Pa] from Nimmo 2015 Table 2
        pc.P_icb = 328e9 # [Pa] from Nimmo 2015 Table 2
        pc.T_c = 4180 # [K] from Nimmo 2015 Table 2
        pc.T_i = 5508 # [K] from Nimmo 2015 Table 2
        pc.T_cen = 5726 # [K] from Nimmo 2015 Table 2
        pc.T_m0 = 2677 # [K] from Nimmo 2015 Table 2
        pc.T_m1 = 2.95e-12 # [ /Pa] from Nimmo 2015 Table 2
        pc.T_m2 = 8.37e-25 # [ /Pa^2] from Nimmo 2015 Table 2
        pc.alpha = 1.25e-5 # [ /K] from Nimmo 2015 Table 2
        pc.L_H = 750e3 # [J/kg] from Nimmo 2015 Table 2
        pc.k = 130 # [W/m-K] from Nimmo 2015 Table 2
        pc.D = 6203e3 # [m] from Nimmo 2015 Table 2
        pc.D_k = 5900e3 # [m] pg. 42 from Nimmo 2015 Table 2
        pc.C_p = 840 # [J/kg-K] from Nimmo 2015 Table 2
        pc.alpha_c = 1.1 # [-] from Nimmo 2015 Table 2
        pc.delta_rho_c = 560 # [kg/m^3] from Nimmo 2015 Table 2
        pc.C_r = -10100 # [m/K] from Nimmo 2015 Table 2
        pc.G = 6.67408e-11 # [m^3/kg-s] from Nimmo 2015
        # pc.h_0 = 1.e-11 # - [W/kg] similar to Stevenson Table I
        pc.h_0 = 1.e-31 # - [W/kg] similar to Stevenson Table I
        pc.lam = 1.38e-17 # - [1/s] from Stevenson Table I
        pc.Khl = 1.251e9 # [yr] half-life of potassium-40
        pc.T_D = 5000.  # K - from Nimmo 2015

        CoreLayer.__init__(self, 0., pc.r_c, params=params)
        self.compute_mass_of_core()
        self.reset_current_values()

    def reset_current_values(self):
        self.current_values = Parameters('current_values')
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
        '''
        density

        :param r: radius to compute [m]
        :return: rho [kg/m^3]
        '''
        p = self.params.core
        return p.rho_cen * exp(-r ** 2 / p.L ** 2)

    def g(self, r):
        '''
        gravity at particular radius

        :param r: radius [m]
        :return: g [m^2/s]
        '''
        p = self.params.core
        return 4 * pi / 3 * p.G * p.rho_cen * r * (1 - 3 * r ** 2 / (5 * p.L ** 2))

    def P(self, r):
        '''
        pressure at particular radius

        :param r: radius [m]
        :return: pressure [Pa]
        '''
        p = self.params.core
        return p.P_c + 4 * pi * p.G * p.rho_cen ** 2 / 3 * (
        (3 * p.r_c ** 2 / 10 - p.L ** 2 / 5) * exp(-p.r_c ** 2 / p.L ** 2) - (3 * r ** 2 / 10 - p.L ** 2 / 5) * exp(
            -r ** 2 / p.L ** 2))

    def T_m(self, P):
        '''
        liquidus temperature at pressure P

        :param P: pressure [Pa]
        :return: T [K]
        '''
        p = self.params.core
        return p.T_m0 * (1 + p.T_m1 * P + p.T_m2 * P ** 2)

    def T_adiabat_from_T_cen(self, T_cen, r):
        p = self.params.core
        return T_cen * exp(-r ** 2 / p.D ** 2)

    def T_adiabat_from_T_cmb(self, T_cmb, r):
        p = self.params.core
        T_cen = self.T_cen_from_T_cmb(T_cmb)
        return T_cen * exp(-r ** 2 / p.D ** 2)

    def dTa_dr(self, T_cmb, r):
        p = self.params.core
        T_cen = self.T_cen_from_T_cmb(T_cmb)
        return T_cen * (-2 * r / p.D ** 2) * exp(-r ** 2 / p.D ** 2)

    def T_cen_from_T_cmb(self, T_cmb):
        p = self.params.core
        return T_cmb * exp(p.r_c ** 2 / p.D ** 2)

    def compute_mass_of_core(self):
        p = self.params.core
        self.mass = self.compute_mass_of_partial_core(p.r_c, 0.)
        return self.mass

    def compute_mass_of_partial_core(self, r_top, r_bottom):
        p = self.params.core
        return 4 * pi * p.rho_cen * (
        (-p.L ** 2 / 2 * r_top * exp(-r_top ** 2 / p.L ** 2) + p.L ** 3 / 4 * pi ** 0.5 * spec.erf(r_top / p.L))
        - (-p.L ** 2 / 2 * r_bottom * exp(-r_bottom ** 2 / p.L ** 2) + p.L ** 3 / 4 * pi ** 0.5 * spec.erf(
            r_bottom / p.L)))

    def C_r(self, T_cmb, r_i=None, recompute=False, store_computed=True):
        '''
        constant relation core growth to temperature change

        :param T_cmb:
        :param r_i:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.C_r is not None and not recompute:
            return self.current_values.C_r
        else:
            dT = 1e-6
            r_i = self.r_i(T_cmb, recompute=True, store_computed=False)
            r_ip = self.r_i(T_cmb + dT, recompute=True, store_computed=False)
            C_r = (r_ip - r_i) / dT
            if store_computed:
                self.current_values.C_r = C_r
            return C_r
        # def C_r(self, T_cmb, r_i=None):
        p = self.params.core
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
        '''
        constant relating light element release to core growth

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.C_c is not None and not recompute:
            return self.current_values.C_c
        else:
            r_i = self.r_i(T_cmb, recompute=recompute, store_computed=store_computed)
            if r_i == p.r_c:
                C_c = 0.
            else:
                M_oc = self.compute_mass_of_partial_core(p.r_c, r_i)
                C_c = 4 * pi * r_i ** 2 * p.delta_rho_c / (M_oc * p.alpha_c)
            if store_computed:
                self.current_values.C_c = C_c
            return C_c

    def I_s(self, T_cmb, recompute=False, store_computed=True):
        '''
        Integral for secular cooling, eq (54) Nimmo 2015

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.I_s is not None and not recompute:
            return self.current_values.I_s
        else:
            T_cen = self.T_cen_from_T_cmb(T_cmb)
            A = (1 / p.L ** 2 + 1 / p.D ** 2) ** -0.5 # eq (55) Nimmo 2015
            I_s = 4 * pi * T_cen * p.rho_cen * (
            -A ** 2 * p.r_c / 2 * exp(-p.r_c ** 2 / A ** 2) + A ** 3 * pi ** 0.5 / 4 * spec.erf(p.r_c / A))
            if store_computed:
                self.current_values.I_s = I_s
            return I_s

    def I_T(self, T_cmb, recompute=False, store_computed=True):
        p = self.params.core
        if self.current_values.I_T is not None and not recompute:
            return self.current_values.I_T
        else:
            T_cen = self.T_cen_from_T_cmb(T_cmb)
            Bsq = (1 / p.L ** 2 - 1 / p.D ** 2) ** -1
            I_T = 4 * pi * p.rho_cen / (3 * T_cen) * p.r_c ** 3 * (1 - 3 * p.r_c ** 2 / (5 * Bsq))
            if store_computed:
                self.current_values.I_T = I_T
            return I_T

    def I_g(self, T_cmb, recompute=False, store_computed=True):
        p = self.params.core
        if self.current_values.I_g is not None and not recompute:
            return self.current_values.I_g
        else:
            r_i = self.r_i(T_cmb, recompute=recompute, store_computed=store_computed)
            Csq = 3 * p.L ** 2 / 16 - p.r_c ** 2 / 2 * (1 - 3 * p.r_c ** 2 / (10 * p.L ** 2))
            I_g = 8 * pi ** 2 * p.rho_cen ** 2 * p.G / 3 * (
                (3 / 20 * p.r_c ** 5 - p.L ** 2 / 8 * p.r_c ** 3 - p.L ** 2 * Csq * p.r_c) * exp(
                    -p.r_c ** 2 / p.L ** 2) + Csq / 2 * p.L ** 3 * pi ** 0.5 * spec.erf(p.r_c / p.L)
                - ((3 / 20 * r_i ** 5 - p.L ** 2 / 8 * r_i ** 3 - p.L ** 2 * Csq * r_i) * exp(
                    -r_i ** 2 / p.L ** 2) + Csq / 2 * p.L ** 3 * pi ** 0.5 * spec.erf(r_i / p.L))
            )
            if store_computed:
                self.current_values.I_g = I_g
            return I_g

    def phi(self, r):
        p = self.params.core
        return (2 / 3 * pi * p.G * p.rho_cen * r ** 2 * (1 - 3 * r ** 2 / (10 * p.L ** 2))
                - (2 / 3 * pi * p.G * p.rho_cen * p.r_c ** 2 * (1 - 3 * p.r_c ** 2 / (10 * p.L ** 2))))

    def dr_i_dt(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        p = self.params.core
        if self.current_values.dr_i_dt is not None and not recompute:
            return self.current_values.dr_i_dt
        else:
            dr_i_dt = self.C_r(T_cmb, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.dr_i_dt = dr_i_dt
            return dr_i_dt

    def Dc_Dt(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        p = self.params.core
        if self.current_values.Dc_Dt is not None and not recompute:
            return self.current_values.Dc_Dt
        else:
            Dc_Dt = self.C_c(T_cmb, recompute=recompute, store_computed=store_computed) * self.C_r(T_cmb,
                                                                                                   recompute=recompute,
                                                                                                   store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Dc_Dt = Dc_Dt
            return

    def compute_Lhp(self, T_cmb, dP=1., recompute=False, store_computed=True):
        p = self.params.core
        # P_icb = self.P(r_i)
        # dTm_dP = (self.T_m(P_icb)-self.T_m(P_icb+dP))/dP
        return p.L_H

    def heat_production_per_kg(self, time):
        p = self.params.core
        '''
        Equation (2) from Stevenson et al 1983
        '''
        return p.h_0 * np.exp(-p.lam * time)

    def T_R(self, T_cmb, h, recompute=False, store_computed=True):
        '''
        Compute T_R, the effective value where T_R = Q_R/E_R from Nimmo 2015 eq (74)

        :param T_cmb:
        :param h:
        :param recompute:
        :param store_computed:
        :return: T_R [K]
        '''
        p = self.params.core
        if self.current_values.T_R is not None and not recompute:
            return self.current_values.T_R
        else:
            if h == 0.:
                T_R = 1e99
            else:
                T_R = (self.Q_R(h, recompute=recompute, store_computed=store_computed)
                       / self.E_R(T_cmb, h, recompute=recompute, store_computed=store_computed))
            if store_computed:
                self.current_values.T_R = T_R
            return T_R

    def r_i(self, T_cmb, recompute=False, store_computed=True, one_off=False):
        p = self.params.core
        if self.current_values.r_i is not None and not recompute and not one_off:
            return self.current_values.r_i
        else:
            TaTm = lambda r: self.T_adiabat_from_T_cmb(T_cmb, r) - self.T_m(self.P(r))
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
        '''
        heat production per kelvin for secular cooling eq (57) Nimmo 2015

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Qt_s is not None and not recompute:
            return self.current_values.Qt_s
        else:
            Qt_s = -p.C_p / T_cmb * self.I_s(T_cmb, recompute=recompute, store_computed=store_computed)
            if store_computed:
                self.current_values.Qt_s = Qt_s
            return Qt_s

    def Q_s(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        '''
        heat production for secular cooling eq (57) Nimmo 2015

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_s is not None and not recompute:
            return self.current_values.Q_s
        else:
            Q_s = -p.C_p / T_cmb * dT_cmb_dt * self.I_s(T_cmb, recompute=recompute, store_computed=store_computed)
            if store_computed:
                self.current_values.Q_s = Q_s
            return Q_s

    def Et_s(self, T_cmb, recompute=False, store_computed=True):
        '''
        entropy production per kelving for secular cooling

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Et_s is not None and not recompute:
            return self.current_values.Et_s
        else:
            Et_s = p.C_p / T_cmb * (
            self.mass - self.I_s(T_cmb, recompute=recompute, store_computed=store_computed) / T_cmb)
            if store_computed:
                self.current_values.Et_s = Et_s
            return Et_s

    def E_s(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        '''
        entropy production for secular cooling

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.E_s is not None and not recompute:
            return self.current_values.E_s
        else:
            E_s = p.C_p / T_cmb * (
            self.mass - self.I_s(T_cmb, recompute=recompute, store_computed=store_computed) / T_cmb) * dT_cmb_dt
            if store_computed:
                self.current_values.E_s = E_s
            return E_s

    def Q_R(self, h, recompute=False, store_computed=True):
        '''
        heat production from radioactive decay

        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_R is not None and not recompute:
            return self.current_values.Q_R
        else:
            Q_R = self.mass * h
            if store_computed:
                self.current_values.Q_R = Q_R
            return Q_R

    def E_R(self, T_cmb, h, recompute=False, store_computed=True):
        '''
        entropy production from radioactive decay

        :param T_cmb:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.E_R is not None and not recompute:
            return self.current_values.E_R
        else:
            E_R = (self.mass / T_cmb - self.I_T(T_cmb, recompute=recompute, store_computed=store_computed)) * h
            if store_computed:
                self.current_values.E_R = E_R
            return E_R

    def Qt_L(self, T_cmb, recompute=False, store_computed=True):
        '''
        heat production per kelvin for latent heat release form inner-core growth

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Qt_L is not None and not recompute:
            return self.current_values.Qt_L
        else:
            r_i = self.r_i(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_L = 4 * pi * r_i ** 2 * self.compute_Lhp(T_cmb, recompute=recompute,
                                                        store_computed=store_computed) * self.rho(r_i) * self.C_r(
                T_cmb, recompute=recompute, store_computed=store_computed)
            if store_computed:
                self.current_values.Qt_L = Qt_L
            return Qt_L

    def Q_L(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        '''
        heat production from latent heat from inner-core growth

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_L is not None and not recompute:
            return self.current_values.Q_L
        else:
            Q_L = self.Qt_L(T_cmb, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_L = Q_L
            return Q_L

    def Et_L(self, T_cmb, recompute=False, store_computed=True):
        '''
        entropy production per kelvin for latent heat from inner core growth

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Et_L is not None and not recompute:
            return self.current_values.Et_L
        else:
            T_i = self.T_adiabat_from_T_cmb(T_cmb,
                                            self.r_i(T_cmb, recompute=recompute, store_computed=store_computed))
            Et_L = self.Qt_L(T_cmb, recompute=recompute, store_computed=store_computed) * (T_i - T_cmb) / (
            T_i * T_cmb)
            if store_computed:
                self.current_values.Et_L = Et_L
            return Et_L

    def E_L(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        '''
        entropy production from latent heat fron inner-core growth

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.E_L is not None and not recompute:
            return self.current_values.E_L
        else:
            E_L = self.Et_L(T_cmb, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.E_L = E_L
            return E_L

    def Qt_g(self, T_cmb, recompute=False, store_computed=True):
        '''
        heat production per kelvin for compositional gravitational convection from inner-core growth

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Qt_g is not None and not recompute:
            return self.current_values.Qt_g
        else:
            M_oc = self.compute_mass_of_partial_core(p.r_c, self.r_i(T_cmb))
            Qt_g = (self.I_g(T_cmb, recompute=recompute, store_computed=store_computed)
                    - M_oc * self.phi(self.r_i(T_cmb, recompute=recompute, store_computed=store_computed))) * (
                p.alpha_c * self.C_c(T_cmb, recompute=recompute, store_computed=store_computed)
                * self.C_r(T_cmb, recompute=recompute, store_computed=store_computed))
            if store_computed:
                self.current_values.Qt_g = Qt_g
            return Qt_g

    def Q_g(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        '''
        heat production for compositional gravitational convection from inner-core growth

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_g is not None and not recompute:
            return self.current_values.Q_g
        else:
            Q_g = self.Qt_g(T_cmb, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_g = Q_g
            return Q_g

    def Et_g(self, T_cmb, recompute=False, store_computed=True):
        '''
        entropy prodution per kelvin for composition gravitational convection from IC growth

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Et_g is not None and not recompute:
            return self.current_values.Et_g
        else:
            Et_g = self.Qt_g(T_cmb, recompute=recompute, store_computed=store_computed) / T_cmb
            if store_computed:
                self.current_values.Et_g = Et_g
            return Et_g

    def E_g(self, T_cmb, dT_cmb_dt, recompute=False, store_computed=True):
        '''
        entropy production from compositional gravitational convection from IC growth

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.E_g is not None and not recompute:
            return self.current_values.E_g
        else:
            E_g = self.Q_g(T_cmb, dT_cmb_dt) / T_cmb
            if store_computed:
                self.current_values.E_g = E_g
            return E_g

    def Q_k(self, T_cmb, recompute=False, store_computed=True):
        '''
        heat conducted down adiabat

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_k is not None and not recompute:
            return self.current_values.Q_k
        else:
            Q_k = 8 * pi * p.r_c ** 3 * p.k * T_cmb / p.D ** 2
            if store_computed:
                self.current_values.Q_k = Q_k
            return Q_k

    def E_k(self, recompute=False, store_computed=True):
        '''
        entropy contribution from heat conducted down adiabat

        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.E_k is not None and not recompute:
            return self.current_values.E_k
        else:
            E_k = 16 * pi * p.k * p.r_c ** 5 / (5 * p.D ** 4)
            if store_computed:
                self.current_values.E_k = E_k
            return E_k
        # def E_k(self):
        p = self.params.core
        #     return 16*pi*p.k*p.r_c**5/(5*p.D**4)*(1+2/(7*p.D_k**2/p.r_c**2-1))

    def Qt_T(self, T_cmb, recompute=False, store_computed=True):
        '''
        total heat flow per kelvin for terms dependent on temperature change

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Qt_T is not None and not recompute:
            return self.current_values.Qt_T
        else:
            Qt_g = self.Qt_g(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_L = self.Qt_L(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_s = self.Qt_s(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_T = Qt_g + Qt_L + Qt_s
            if store_computed:
                self.current_values.Qt_T = Qt_T
            return Qt_T

    def Et_T(self, T_cmb, recompute=False, store_computed=True):
        '''
        total entropy per kelvin for terms dependent on temperature change

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Et_T is not None and not recompute:
            return self.current_values.Et_T
        else:
            Et_g = self.Et_g(T_cmb, recompute=recompute, store_computed=store_computed)
            Et_L = self.Et_L(T_cmb, recompute=recompute, store_computed=store_computed)
            Et_s = self.Et_s(T_cmb, recompute=recompute, store_computed=store_computed)
            Et_T = Et_g + Et_L + Et_s
            if store_computed:
                self.current_values.Et_T = Et_T
            return Et_T

    def Q_cmb(self, T_cmb, dT_cmb_dt, h, recompute=False, store_computed=True):
        '''
        total heat flow at CMB

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_cmb is not None and not recompute:
            return self.current_values.Q_cmb
        else:
            Q_R = self.Q_R(h, recompute=recompute, store_computed=store_computed)
            Qt_T = self.Qt_T(T_cmb, recompute=recompute, store_computed=store_computed)
            Q_cmb = Q_R + Qt_T * dT_cmb_dt
            if store_computed:
                self.current_values.Q_cmb = Q_cmb
            return Q_cmb

    def Delta_E(self, T_cmb, dT_cmb_dt, h, recompute=False, store_computed=True):
        '''
        total entropy balance

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Delta_E is not None and not recompute:
            return self.current_values.Delta_E
        else:
            E_R = self.E_R(T_cmb, h, recompute=recompute, store_computed=store_computed)
            Et_T = self.Et_T(T_cmb, recompute=recompute, store_computed=store_computed)
            E_k = self.E_k(recompute=recompute, store_computed=store_computed)
            Delta_E = E_R + Et_T * dT_cmb_dt - E_k
            if store_computed:
                self.current_values.Delta_E = Delta_E
            return Delta_E

    def Q_phi(self, T_cmb, dT_cmb_dt, h, T_D, recompute=False, store_computed=True):
        '''
        heat prodution rate powering dynamo

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param T_D:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_phi is not None and not recompute:
            return self.current_values.Q_phi
        else:
            E_phi = self.E_phi(T_cmb, dT_cmb_dt, h, recompute=recompute, store_computed=store_computed)
            Q_phi = E_phi * T_D
            if store_computed:
                self.current_values.Q_phi = Q_phi
            return Q_phi

    def E_phi(self, T_cmb, dT_cmb_dt, h, recompute=False, store_computed=True):
        '''
        entropy production rate powering dynamo

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.E_phi is not None and not recompute:
            return self.current_values.E_phi
        else:
            Et_T = self.Et_T(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_T = self.Qt_T(T_cmb, recompute=recompute, store_computed=store_computed)
            T_R = self.T_R(T_cmb, h, recompute=recompute, store_computed=store_computed)
            Q_cmb = self.Q_cmb(T_cmb, dT_cmb_dt, h, recompute=recompute, store_computed=store_computed)
            Q_R = self.Q_R(h, recompute=recompute, store_computed=store_computed)
            E_k = self.E_k(recompute=recompute, store_computed=store_computed)
            E_phi = (Q_cmb - Q_R * (1 - Qt_T / Et_T / T_R)) * Et_T / Qt_T - E_k
            if store_computed:
                self.current_values.E_phi = E_phi
            return E_phi

    def Q_adiabat_at_r(self, T_cmb, r):
        '''
        heat flow down adiabat at particular radius

        :param T_cmb:
        :param r:
        :return:
        '''
        p = self.params.core
        Q_adiabat = 8 * pi * p.k * r ** 3 / p.D ** 2 * self.T_adiabat_from_T_cmb(T_cmb, r)
        return Q_adiabat

    def stable_layer_thickness(self, T_cmb, dT_cmb_dt, h, recompute=False, store_computed=True):
        '''
        distance below CMB where heat flow down adiabat matches heat flow across CMB

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        Q_cmb = self.Q_cmb(T_cmb, dT_cmb_dt, h, recompute=True, store_computed=False)

        D_stable = lambda r: self.Q_adiabat_at_r(T_cmb, r) - Q_cmb
        if Q_cmb > self.Q_adiabat_at_r(T_cmb, p.r_c):
            return 0.
        elif Q_cmb < 0.:
            return p.r_c
        else:
            return p.r_c - opt.brentq(D_stable, p.r_c, 0.)

    def energy_balance(self, time, T_cmb, q_cmb_flux):
        '''
        Compute dT_cmb/dt given the current time, T_cmb, and heat flux at CMB

        :param time: time from formation in [s]
        :param T_cmb: CMB temperature in [K]
        :param q_cmb_flux: CMB heat flow in [W/m^2]
        :return: dT_cmb_dt: change in T_cmb with time [K/s]
        '''
        p = self.params.core
        self.reset_current_values()
        Qt_T = self.Qt_T(T_cmb)
        Q_R = self.Q_R(self.heat_production_per_kg(time))
        Q_cmb = q_cmb_flux * self.outer_surface_area
        dT_cmb_dt = (Q_cmb - Q_R) / Qt_T
        return dT_cmb_dt

class MantleLayer_Driscoll(MantleLayer):
    pass

class Radiogenics(object):
    def __init__(self):
        self.CGyr2s = 1e9 * 365.25 * 24 * 3600
        self.l_238U = 4.9160e-18  # [1/Gyr > 1/s]
        self.l_235U = 3.1209e-17  # [1/Gyr > 1/s]
        self.l_232Th = 1.5633e-18  # [1/Gyr > 1/s]
        self.l_40K = 1.7558e-17  # [1/Gyr > 1/s]

    def heat_production_mantle(self, Hp, t, tp=1.44155e17):
        '''
        Calculates heat production through time given present-day heat production Hp

        :param Hp: heat-production at current day [W/kg, TW, or W/m^3]
        :param t: time [s]
        :param tp: end-time [s] default: 1.44155e17 s (4.568 Byr)
        :return: Hm(t): heat-production at time t, in same units as provided in Hp
        '''
        h_238U = 0.37196  # normalized W/kg Koreneaga 2005
        h_235U = 0.01638  # normalized W/kg Koreneaga 2005
        h_232Th = 0.43028  # normalized W/kg Koreneaga 2005
        h_40K = 0.18137  # normalized W/kg Koreneaga 2005
        return Hp * (self._heat_hp(h_238U, self.l_238U, t, tp=tp)
                     + self._heat_hp(h_235U, self.l_235U, t, tp=tp)
                     + self._heat_hp(h_232Th, self.l_232Th, t, tp=tp)
                     + self._heat_hp(h_40K, self.l_40K, t, tp=tp))

    def heat_production_core(self, Hp, t, tp=1.44155e17):
        '''
        Calculates heat production through time given present-day heat production

        :param Hp: heat production at present day in [W, W/kg, W/m^3]
        :param t: time [s]
        :param tp: present-day time [s] default 1.44155e17 s (4.568 Byr)
        :return: Hc(t) heat production in core same units as provided in Hp
        '''
        h_40K = 1.  # normalized W/kg
        return Hp * self._heat_hp(h_40K, self.l_40K, t, tp=tp)

    def _heat_h0(self, h0, lam, t):
        return h0 * np.exp(-lam * t)

    def _heat_hp(self, hp, lam, t, tp=1.44155e17):
        return hp * np.exp(lam * (tp - t))

    def ppm2Hp_40K_Driscoll(self, ppm):
        '''
        Converts 40K ppm to present-day heat-production in W, using numbers from Nimmo 2004

        :param ppm:
        :return:
        '''
        return 2e12 / 255 * ppm

    def ppm2Hp_40K_Nimmo(self, ppm):
        '''
        Converts 40K ppm to present-day heat-production in W, using numbers from Driscoll 2014

        :param ppm:
        :return:
        '''
        return 2.1e12 / 300 * ppm

    def ppm2Hp_40K_Driscoll(self, Hp):
        '''
        Converts present-day heat-production in W to 40K ppm, using numbers from Driscoll 2014

        :param Hp:
        :return:
        '''
        return 255/2e12* Hp

    def Hp2ppm_Nimmo(self, Hp):
        '''
        Converts present-day heat-production in W to 40K ppm, using numbers from Nimmo 2004

        :param Hp:
        :return:
        '''
        return 300/2.1e12* Hp

class CoreLayer_Custom(CoreLayer_Nimmo):
    def __init__(self, params=None):
        CoreLayer_Nimmo.__init__(self, params)
        pc = self.params.core
        pc.Hp = 1e12 # [W] heat production from radiogenics at present day
        pc.alpha_cm = 0. # [-] coefficient of compositional expansion for MgO [O'Rourke, Korenaga 2016]
        pc.alpha_cs = 0. # [-] coefficient of compositional expansion for SiO2 [Hirose et al. 2017]
        pc.alpha_cf = 0. # [-] coefficient of compositional expansion for FeO (guess)
        pc.L_Hm = 0. # [J/kg] latent heat for MgO exsolution (guess)
        pc.L_Hs = 0. # [J/kg] latent heat for SiO2 exsolution [Hirose et al. 2017]
        pc.L_Hf = 0. # [J/kg] latent heat for FeO exsolution (guess)
        # pc.alpha_cm = 0.8 # [-] coefficient of compositional expansion for MgO [O'Rourke, Korenaga 2016]
        # pc.alpha_cs = 1.117 # [-] coefficient of compositional expansion for SiO2 [Hirose et al. 2017]
        # pc.alpha_cf = 0.5 # [-] coefficient of compositional expansion for FeO (guess)
        # pc.L_Hm = 910e3 # [J/kg] latent heat for MgO exsolution (guess)
        # pc.L_Hs = 4.3e6 # [J/kg] latent heat for SiO2 exsolution [Hirose et al. 2017]
        # pc.L_Hf = 910e3 # [J/kg] latent heat for FeO exsolution (guess)

    def reset_current_values(self):
        '''
        overloaded method to reset new values
        '''
        self.current_values = Parameters('current_values')
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
        # added values:
        self.current_values.C_m = None
        self.current_values.Qt_gm = None
        self.current_values.Q_gm = None
        self.current_values.Et_gm = None
        self.current_values.E_gm = None
        self.current_values.Qt_Lm = None
        self.current_values.Q_Lm = None
        self.current_values.C_s = None
        self.current_values.Qt_gs = None
        self.current_values.Q_gs = None
        self.current_values.Et_gs = None
        self.current_values.E_gs = None
        self.current_values.Qt_Ls = None
        self.current_values.Q_Ls = None
        self.current_values.C_f = None
        self.current_values.Qt_gf = None
        self.current_values.Q_gf = None
        self.current_values.Et_gf = None
        self.current_values.E_gf = None
        self.current_values.Qt_Lf = None
        self.current_values.Q_Lf = None
        self.current_values.Moles = None
        self.current_values.dMoles_dT = None
        self.current_values.dKs_dT = None

    def heat_production_per_kg(self, time):
        '''
        Overloaded method to use custom radiogenics package

        :param time: time [s]
        :return: heat production [W/kg]
        '''
        pc = self.params.core
        return self.planet.radiogenics.heat_production_core(pc.Hp, time)/self.mass

    def C_m(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        constant relating MgO exsolution to CMB temperature change [wt% / K]

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.C_m is not None and not recompute:
            return self.current_values.C_m
        else:
            pr = self.params.reactions
            # compute dKs given T_cmb and Moles
            if self.current_values.dKs_dT is not None and not recompute:
                dKs_dT = self.current_values.dKs_dT
            else:
                dKs_dT = self.planet.reactions.compute_dKs_dT(T_cmb, Moles)
                if store_computed:
                    self.current_values.dKs_dT = dKs_dT

            # compute dMoles_dT given T_cmb, Moles, and dKs
            if self.current_values.dMoles_dT is not None and not recompute:
                dMoles_dT = self.current_values.dMoles_dT
            else:
                dMoles_dT = self.planet.reactions.dMoles_dT(Moles, T_cmb, dKs_dT=dKs_dT, dTdt=1e12) #HACK for erosion
                # if store_computed:
                    # self.current_values.dMoles_dT = dMoles_dT

            # compute C_m dependent on solubility of X_Mg compared to current X_Mg
            # 0 if X_Mg_sol > X_Mg, convert to wt% MgO if X_Mg_sol < X_Mg
            _,_,[C_m, C_s, C_f] = self.planet.reactions.Moles2wtp(dMoles_dT, calculate_Cs=True)
            if store_computed:
                self.current_values.C_m = C_m
                self.current_values.C_s = C_s
                self.current_values.C_f = C_f
            return C_m

    def C_s(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        constant relating SiO2 exsolution to CMB temperature change [wt% / K]

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.C_s is not None and not recompute:
            return self.current_values.C_s
        else:
            pr = self.params.reactions
            # compute dKs given T_cmb and Moles
            if self.current_values.dKs_dT is not None and not recompute:
                dKs_dT = self.current_values.dKs_dT
            else:
                dKs_dT = self.planet.reactions.compute_dKs_dT(T_cmb, Moles)
                if store_computed:
                    self.current_values.dKs_dT = dKs_dT

            # compute dMoles_dT given T_cmb, Moles, and dKs
            if self.current_values.dMoles_dT is not None and not recompute:
                dMoles_dT = self.current_values.dMoles_dT
            else:
                dMoles_dT = self.planet.reactions.dMoles_dT(Moles, T_cmb, dKs_dT=dKs_dT, dTdt=1e12) #HACK for erosion
                # if store_computed:
                    # self.current_values.dMoles_dT = dMoles_dT

            # compute C_m dependent on solubility of X_Mg compared to current X_Mg
            # 0 if X_Mg_sol > X_Mg, convert to wt% MgO if X_Mg_sol < X_Mg
            _, _, [C_m, C_s, C_f] = self.planet.reactions.Moles2wtp(dMoles_dT, calculate_Cs=True)
            if store_computed:
                self.current_values.C_m = C_m
                self.current_values.C_s = C_s
                self.current_values.C_f = C_f
            return C_s

    def C_f(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        constant relating FeO exsolution to CMB temperature change [wt% / K]

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.C_f is not None and not recompute:
            return self.current_values.C_f
        else:
            pr = self.params.reactions
            # compute dKs given T_cmb and Moles
            if self.current_values.dKs_dT is not None and not recompute:
                dKs_dT = self.current_values.dKs_dT
            else:
                dKs_dT = self.planet.reactions.compute_dKs_dT(T_cmb, Moles)
                if store_computed:
                    self.current_values.dKs_dT = dKs_dT

            # compute dMoles_dT given T_cmb, Moles, and dKs
            if self.current_values.dMoles_dT is not None and not recompute:
                dMoles_dT = self.current_values.dMoles_dT
            else:
                dMoles_dT = self.planet.reactions.dMoles_dT(Moles, T_cmb, dKs_dT=dKs_dT, dTdt=1e12) #HACK for erosion
                # if store_computed:
                    # self.current_values.dMoles_dT = dMoles_dT

            # compute C_m dependent on solubility of X_Mg compared to current X_Mg
            # 0 if X_Mg_sol > X_Mg, convert to wt% MgO if X_Mg_sol < X_Mg
            _, _, [C_m, C_s, C_f] = self.planet.reactions.Moles2wtp(dMoles_dT, calculate_Cs=True)
            if store_computed:
                self.current_values.C_m = C_m
                self.current_values.C_s = C_s
                self.current_values.C_f = C_f
            return C_f

    def Qt_gm(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        heat production per kelvin for compositional gravitational convection from MgO exsolution

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Qt_gm is not None and not recompute:
            return self.current_values.Qt_gm
        else:
            M_oc = self.compute_mass_of_partial_core(pc.r_c, self.r_i(T_cmb))
            Qt_gm = (self.I_g(T_cmb, recompute=recompute, store_computed=store_computed)
                    - M_oc * self.phi(self.r_i(T_cmb, recompute=recompute, store_computed=store_computed))) * (
                pc.alpha_cm * self.C_m(T_cmb, Moles, recompute=recompute, store_computed=store_computed))
            if store_computed:
                self.current_values.Qt_gm = Qt_gm
            return Qt_gm

    def Qt_gs(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        heat production per kelvin for compositional gravitational convection from SiO2 exsolution

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Qt_gs is not None and not recompute:
            return self.current_values.Qt_gs
        else:
            M_oc = self.compute_mass_of_partial_core(pc.r_c, self.r_i(T_cmb))
            Qt_gs = (self.I_g(T_cmb, recompute=recompute, store_computed=store_computed)
                    - M_oc * self.phi(self.r_i(T_cmb, recompute=recompute, store_computed=store_computed))) * (
                pc.alpha_cs * self.C_s(T_cmb, Moles, recompute=recompute, store_computed=store_computed))
            if store_computed:
                self.current_values.Qt_gs = Qt_gs
            return Qt_gs

    def Qt_gf(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        heat production per kelvin for compositional gravitational convection from FeO exsolution

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Qt_gf is not None and not recompute:
            return self.current_values.Qt_gf
        else:
            M_oc = self.compute_mass_of_partial_core(pc.r_c, self.r_i(T_cmb))
            Qt_gf = (self.I_g(T_cmb, recompute=recompute, store_computed=store_computed)
                    - M_oc * self.phi(self.r_i(T_cmb, recompute=recompute, store_computed=store_computed))) * (
                pc.alpha_cf * self.C_f(T_cmb, Moles, recompute=recompute, store_computed=store_computed))
            if store_computed:
                self.current_values.Qt_gf = Qt_gf
            return Qt_gf

    def Q_gm(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        heat production for compositional gravitational convection from MgO exsolution

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Q_gm is not None and not recompute:
            return self.current_values.Q_gm
        else:
            Q_gm = self.Qt_gm(T_cmb, Moles, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_gm = Q_gm
            return Q_gm

    def Q_gs(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        heat production for compositional gravitational convection from SiO2 exsolution

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Q_gs is not None and not recompute:
            return self.current_values.Q_gs
        else:
            Q_gs = self.Qt_gs(T_cmb, Moles, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_gs = Q_gs
            return Q_gs

    def Q_gf(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        heat production for compositional gravitational convection from FeO exsolution

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Q_gf is not None and not recompute:
            return self.current_values.Q_gf
        else:
            Q_gf = self.Qt_gf(T_cmb, Moles, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_gf = Q_gf
            return Q_gf

    def Et_gm(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        entropy prodution per kelvin for composition gravitational convection from Mg exsolution

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Et_gm is not None and not recompute:
            return self.current_values.Et_gm
        else:
            Et_gm = self.Qt_gm(T_cmb, Moles, recompute=recompute, store_computed=store_computed) / T_cmb
            if store_computed:
                self.current_values.Et_gm = Et_gm
            return Et_gm

    def Et_gs(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        entropy prodution per kelvin for composition gravitational convection from Mg exsolution

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Et_gs is not None and not recompute:
            return self.current_values.Et_gs
        else:
            Et_gs = self.Qt_gs(T_cmb, Moles, recompute=recompute, store_computed=store_computed) / T_cmb
            if store_computed:
                self.current_values.Et_gs = Et_gs
            return Et_gs

    def Et_gf(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        entropy prodution per kelvin for composition gravitational convection from Mg exsolution

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Et_gf is not None and not recompute:
            return self.current_values.Et_gf
        else:
            Et_gf = self.Qt_gf(T_cmb, Moles, recompute=recompute, store_computed=store_computed) / T_cmb
            if store_computed:
                self.current_values.Et_gf = Et_gf
            return Et_gf

    def E_gm(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        entropy production from compositional gravitational convection from Mg exsolution

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.E_gm is not None and not recompute:
            return self.current_values.E_gm
        else:
            E_gm = self.Q_gm(T_cmb, dT_cmb_dt, XMg=XMg, XO=XO, C=C) / T_cmb
            if store_computed:
                self.current_values.E_gm = E_gm
            return E_gm

    def E_gs(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        entropy production from compositional gravitational convection from Mg exsolution

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.E_gs is not None and not recompute:
            return self.current_values.E_gs
        else:
            E_gs = self.Q_gs(T_cmb, dT_cmb_dt, XMg=XMg, XO=XO, C=C) / T_cmb
            if store_computed:
                self.current_values.E_gs = E_gs
            return E_gs

    def E_gf(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        entropy production from compositional gravitational convection from Mg exsolution

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.E_gf is not None and not recompute:
            return self.current_values.E_gf
        else:
            E_gf = self.Q_gf(T_cmb, dT_cmb_dt, XMg=XMg, XO=XO, C=C) / T_cmb
            if store_computed:
                self.current_values.E_gf = E_gf
            return E_gf

    def Qt_Lm(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        heat production per kelvin for latent heat release from MgO precipitation

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Qt_Lm is not None and not recompute:
            return self.current_values.Qt_Lm
        else:
            C_m = self.C_m(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_Lm = C_m*pc.L_Hm*self.mass
            if store_computed:
                self.current_values.Qt_Lm = Qt_Lm
            return Qt_Lm

    def Qt_Ls(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        heat production per kelvin for latent heat release from MgO precipitation

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Qt_Ls is not None and not recompute:
            return self.current_values.Qt_Ls
        else:
            C_s = self.C_s(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_Ls = C_s*pc.L_Hs*self.mass
            if store_computed:
                self.current_values.Qt_Ls = Qt_Ls
            return Qt_Ls

    def Qt_Lf(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        heat production per kelvin for latent heat release from MgO precipitation

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Qt_Lf is not None and not recompute:
            return self.current_values.Qt_Lf
        else:
            C_f = self.C_f(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_Lf = C_f*pc.L_Hf*self.mass
            if store_computed:
                self.current_values.Qt_Lf = Qt_Lf
            return Qt_Lf

    def Q_Lm(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        heat production from latent heat from MgO precipitation

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_Lm is not None and not recompute:
            return self.current_values.Q_Lm
        else:
            Q_Lm = self.Qt_Lm(T_cmb, Moles, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_Lm = Q_Lm
            return Q_Lm

    def Q_Ls(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        heat production from latent heat from MgO precipitation

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_Ls is not None and not recompute:
            return self.current_values.Q_Ls
        else:
            Q_Ls = self.Qt_Ls(T_cmb, Moles, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_Ls = Q_Ls
            return Q_Ls

    def Q_Lf(self, T_cmb, dT_cmb_dt, Moles, recompute=False, store_computed=True):
        '''
        heat production from latent heat from MgO precipitation

        :param T_cmb:
        :param dT_cmb_dt:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_Lf is not None and not recompute:
            return self.current_values.Q_Lf
        else:
            Q_Lf = self.Qt_Lf(T_cmb, Moles, recompute=recompute, store_computed=store_computed) * dT_cmb_dt
            if store_computed:
                self.current_values.Q_Lf = Q_Lf
            return Q_Lf

    def Qt_T(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        total heat flow per kelvin for terms dependent on temperature change

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Qt_T is not None and not recompute:
            return self.current_values.Qt_T
        else:
            Qt_g = self.Qt_g(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_L = self.Qt_L(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_s = self.Qt_s(T_cmb, recompute=recompute, store_computed=store_computed)
            Qt_gm = self.Qt_gm(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_Lm = self.Qt_Lm(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_gs = self.Qt_gs(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_Ls = self.Qt_Ls(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_gf = self.Qt_gf(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_Lf = self.Qt_Lf(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_T = Qt_g + Qt_L + Qt_s + Qt_gm + Qt_Lm + Qt_gs + Qt_Ls + Qt_gf + Qt_Lf
            if store_computed:
                self.current_values.Qt_T = Qt_T
            return Qt_T

    def Et_T(self, T_cmb, Moles, recompute=False, store_computed=True):
        '''
        total entropy per kelvin for terms dependent on temperature change

        :param T_cmb:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Et_T is not None and not recompute:
            return self.current_values.Et_T
        else:
            Et_g = self.Et_g(T_cmb, recompute=recompute, store_computed=store_computed)
            Et_L = self.Et_L(T_cmb, recompute=recompute, store_computed=store_computed)
            Et_s = self.Et_s(T_cmb, recompute=recompute, store_computed=store_computed)
            Et_gm = self.Et_gm(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Et_gs = self.Et_gs(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Et_gf = self.Et_gf(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Et_T = Et_g + Et_L + Et_s + Et_gm
            if store_computed:
                self.current_values.Et_T = Et_T
            return Et_T

    def Q_cmb(self, T_cmb, dT_cmb_dt, h, Moles, recompute=False, store_computed=True):
        '''
        total heat flow at CMB

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Q_cmb is not None and not recompute:
            return self.current_values.Q_cmb
        else:
            Q_R = self.Q_R(h, recompute=recompute, store_computed=store_computed)
            Qt_T = self.Qt_T(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Q_cmb = Q_R + Qt_T * dT_cmb_dt
            if store_computed:
                self.current_values.Q_cmb = Q_cmb
            return Q_cmb

    def Delta_E(self, T_cmb, dT_cmb_dt, h, Moles, recompute=False, store_computed=True):
        '''
        total entropy balance

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        p = self.params.core
        if self.current_values.Delta_E is not None and not recompute:
            return self.current_values.Delta_E
        else:
            E_R = self.E_R(T_cmb, h, recompute=recompute, store_computed=store_computed)
            Et_T = self.Et_T(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            E_k = self.E_k(recompute=recompute, store_computed=store_computed)
            Delta_E = E_R + Et_T * dT_cmb_dt - E_k
            if store_computed:
                self.current_values.Delta_E = Delta_E
            return Delta_E

    def Q_phi(self, T_cmb, dT_cmb_dt, h, Moles, recompute=False, store_computed=True):
        '''
        heat prodution rate powering dynamo

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param T_D:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.Q_phi is not None and not recompute:
            return self.current_values.Q_phi
        else:
            E_phi = self.E_phi(T_cmb, dT_cmb_dt, h, Moles, recompute=recompute, store_computed=store_computed)
            Q_phi = E_phi * pc.T_D
            if store_computed:
                self.current_values.Q_phi = Q_phi
            return Q_phi

    def E_phi(self, T_cmb, dT_cmb_dt, h, Moles, recompute=False, store_computed=True):
        '''
        entropy production rate powering dynamo

        :param T_cmb:
        :param dT_cmb_dt:
        :param h:
        :param recompute:
        :param store_computed:
        :return:
        '''
        pc = self.params.core
        if self.current_values.E_phi is not None and not recompute:
            return self.current_values.E_phi
        else:
            Et_T = self.Et_T(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            Qt_T = self.Qt_T(T_cmb, Moles, recompute=recompute, store_computed=store_computed)
            T_R = self.T_R(T_cmb, h, recompute=recompute, store_computed=store_computed)
            Q_cmb = self.Q_cmb(T_cmb, dT_cmb_dt, h, Moles, recompute=recompute, store_computed=store_computed)
            Q_R = self.Q_R(h, recompute=recompute, store_computed=store_computed)
            E_k = self.E_k(recompute=recompute, store_computed=store_computed)
            E_phi = (Q_cmb - Q_R * (1 - Qt_T / Et_T / T_R)) * Et_T / Qt_T - E_k
            if store_computed:
                self.current_values.E_phi = E_phi
            return E_phi

    def energy_balance(self, time, T_cmb, q_cmb_flux, Moles):
        '''
        Compute dT_cmb/dt given the current time, T_cmb, and heat flux at CMB

        :param time: time from formation in [s]
        :param T_cmb: CMB temperature in [K]
        :param q_cmb_flux: CMB heat flow in [W/m^2]
        :return: dT_cmb_dt: change in T_cmb with time [K/s]
        '''
        pc = self.params.core
        self.reset_current_values()
        Qt_T = self.Qt_T(T_cmb, Moles)
        Q_R = self.Q_R(self.heat_production_per_kg(time))
        Q_cmb = q_cmb_flux * self.outer_surface_area
        dT_cmb_dt = (Q_cmb - Q_R) / Qt_T
        return dT_cmb_dt

class MantleLayer_Custom(MantleLayer_Stevenson):
    def __init__(self, params=None):
        MantleLayer_Stevenson.__init__(self, params=params)
        pm = self.params.mantle
        pm.Hp = 10e12 # [W] current heat production in mantle

    def heat_production(self, time):
        '''
        Overloaded method to use four-component radiogenics for mantle

        :param time: time [s]
        :return: heat production [W/m^3]
        '''
        pm = self.params.mantle
        return self.planet.radiogenics.heat_production_mantle(pm.Hp, time)/self.volume

class Planet_Custom(Planet):
    def __init__(self, case=1):
        params = Parameters('Custom Stevenson Mantle, Custom Nimmo Core')
        Planet.__init__(self, [CoreLayer_Custom(params=params), MantleLayer_Custom(params=params)], params)
        self.core_layer = self.layers[0]
        self.mantle_layer = self.layers[1]
        self.reactions = sol.Reactions(params=params)
        self.radiogenics = Radiogenics()

    def ODE(self, x, t):
        T_cmb = x[0]
        T_um = x[1]
        Moles = x[2:]
        try:
            self.params.mantle.verbose
            vm = True
        except:
            vm = False
        dTm_dt = self.mantle_layer.energy_balance(T_cmb, T_um, t, verbose=vm)
        cmb_flux = self.mantle_layer.lower_boundary_flux(T_cmb, T_um)
        dTc_dt = self.core_layer.energy_balance(t, T_cmb, cmb_flux, Moles)
        dMoles_dt  = self.reactions.dMoles_dt(Moles=Moles, T_cmb=T_cmb, dTc_dt=dTc_dt)
        dM_Mg_dt, dM_Si_dt, dM_Fe_dt, dM_O_dt, dM_c_dt, dM_MgO_dt, dM_SiO2_dt, dM_FeO_dt, dM_MgSiO3_dt, dM_FeSiO3_dt, dM_m_dt = self.reactions.unwrap_Moles(dMoles_dt)

        return np.array([dTc_dt, dTm_dt, dM_Mg_dt, dM_Si_dt, dM_Fe_dt, dM_O_dt, dM_c_dt, dM_MgO_dt, dM_SiO2_dt, dM_FeO_dt, dM_MgSiO3_dt, dM_FeSiO3_dt, dM_m_dt])

    def integrate(self, times, x0, full_output=False):
        solution = integrate.odeint(self.ODE, x0, times, full_output=full_output)
        return solution

class Mantle_Korenaga2005(MantleLayer):
    def __init__(self):
        def __init__(self, params=None):
            if params is None:
                params = Parameters('Korenaga 2005 for mantle')
            self.params = params
            self.params.mantle = Parameters('Korenaga 2005 for mantle')
            pm = self.params.mantle
            pm.R_c0 = 3480e3 # [m] core radius
            pm.R_p0 = 6372e3 # [m] planet radius
            pm.C = 7e27 # [J/K] heat capacity of whole earth (Korenaga 2005, below eq (1))
            pm.Up = 0.3 # Urey ratio at present day (Korenaga 2005)
            pm.E = 300e3  # [J/mol] activation enthalpy in lower mantle
            pm.T_off = 273 # [K] temperature offset for mantle from surface
            pm.R = 8.314  # [J/mol-K] universal gas constant
            pm.Q0 = 36e12 # [W] mantle heat flow at reference mantle temperature T0
            pm.T0 = 1350 # [K] mantle reference temperature at present day
            pm.beta = 0.3 # [-] exponent for heat flow calculations Nu ~ Ra^beta

            # pm.rho_0 =
            # pm.g =
            # pm.D =
            # pm.h =
            # pm.nu_M =
            # pm.nu_L =
            pm.RadC = 200e3 # [m] radius of curvature of plate bending

            pm.CGyr2s = 1e9 * 365.25 * 24 * 3600
            pm.l_238U = 4.9160e-18  # [1/Gyr > 1/s]
            pm.l_235U = 3.1209e-17  # [1/Gyr > 1/s]
            pm.l_232Th = 1.5633e-18  # [1/Gyr > 1/s]
            pm.l_40K = 1.7558e-17  # [1/Gyr > 1/s]

            MantleLayer.__init__(self, pm.R_c0, pm.R_p0, params)

    def nu(self, T):
        '''
        Calculate non-normalized mantle viscosity eq (10) Korenaga 2005

        :param T:
        :param E:
        :return:
        '''
        pm = self.params.mantle
        return np.exp(pm.E / (pm.R * (T + pm.T_off)))

    def Q_traditional(self, T):
        '''
        Calculates mantle heat flow Q [W] from mantle tempearture T [K] eq (9) Korenaga 2005

        :param T:
        :return:
        '''
        C = C0(pm.Q0, pm.T0, pm.E)
        return C * T * (T / nu(T)) ** beta

    def find_Q0p(self):
        '''
        finds the constant 'a' in eq (19) Korenaga 2005

        :return:
        '''
        pm = self.params.mantle
        Q = self.Q_plates(pm.T0)
        pm.a = pm.Q0/Q
        return pm.a

    def Q_plates(self, T):
        '''
        Calculates surface heat flow scaling with temperature from plate tectonic derviation, eq (19) Korenaga 2005

        :param T:
        :return:
        '''
        Q = pm.a*(pm.C_pe*pm.alpha*pm.rho_0*pm.g*pm.D*pm.h*T**3
                  / (pm.CM_vd*pm.nu_M + pm.CS_vd*pm.nu_L*(pm.h/pm.RadC)**3))**0.5
        return Q

    def C0(self):
        '''
        Calculates the constant relating mantle temperature to heat flow Q, eq (9, 10) Korenaga 2005

        :return:
        '''
        pm = self.params.mantle
        C0 = pm.Q0*pm.T0**(-1-pm.beta)*np.exp(pm.beta*pm.E/(pm.R*(pm.T0+pm.Toff)))
        return C0

    def heat_production_mantle(self, Hp, t, tp=1.44155e17):
        '''
        Calculates heat production through time given present-day heat production Hp

        :param Hp: heat-production at current day [W/kg, TW, or W/m^3]
        :param t: time [s]
        :param tp: end-time [s] default: 1.44155e17 s (4.568 Byr)
        :return: Hm(t): heat-production at time t, in same units as provided in Hp
        '''
        h_238U = 0.37196  # normalized W/kg Koreneaga 2005
        h_235U = 0.01638  # normalized W/kg Koreneaga 2005
        h_232Th = 0.43028  # normalized W/kg Koreneaga 2005
        h_40K = 0.18137  # normalized W/kg Koreneaga 2005
        pm = self.params.mantle

        return Hp * (self._heat_hp(h_238U, pm.l_238U, t, tp=tp)
                     + self._heat_hp(h_235U, pm.l_235U, t, tp=tp)
                     + self._heat_hp(h_232Th, pm.l_232Th, t, tp=tp)
                     + self._heat_hp(h_40K, pm.l_40K, t, tp=tp))

    def heat_production_core(self, Hp, t, tp=1.44155e17):
        '''
        Calculates heat production through time given present-day heat production

        :param Hp: heat production at present day in [W, W/kg, W/m^3]
        :param t: time [s]
        :param tp: present-day time [s] default 1.44155e17 s (4.568 Byr)
        :return: Hc(t) heat production in core same units as provided in Hp
        '''
        pm = self.params.mantle
        h_40K = 1.  # normalized W/kg
        return Hp * self._heat_hp(h_40K, pm.l_40K, t, tp=tp)

    def _heat_h0(self, h0, lam, t):
        '''
        Calculates heat production through time from heat at time 0

        :param h0:
        :param lam:
        :param t:
        :return:
        '''
        return h0 * np.exp(-lam * t)

    def _heat_hp(self, hp, lam, t, tp=1.44155e17):
        '''
        Calculates heat production through time from heat at present time

        :param hp:
        :param lam:
        :param t:
        :param tp:
        :return:
        '''
        return hp * np.exp(lam * (tp - t))

    def energy_balance(self, time, T_mantle, ):
        pass

class Mantle_ORourke2016(MantleLayer):
    def __init__(self, params=None):
        if params is None:
            params = Parameters('ORourke 2016 for mantle')
        self.params = params
        self.params.mantle = Parameters('ORourke 2016 for mantle')
        pm = self.params.mantle
        pm.Heff = 300e3 # [J/mol] activation enthalpy in lower mantle
        pm.T_ref = 2500 # [K] reference temperature for lower mantle
        pm.nu_ref = 2e21 # [Pa-s] reference viscosity for lower mantle
        pm.R = 8.314 # [J/mol-K] universal gas constant

        MantleLayer.__init__(self, pm.R_c0, pm.R_p0, params)

    def nu(self, T):
        '''
        Calculates viscosity in Pa-s of lower mantle given average bondary layer temperature

        :param T: average boundary layer temperature [K]
        :return: nu [Pa-s]
        '''

        pm = self.params.mantle
        return pm.nu_ref*np.exp(pm.Heff/pm.R*(1/T-1/pm.T_ref))

    def Q_M(self, T_M):
        pass
        

    def energy_balance(self, T_M, T_B, T_U):
        pass