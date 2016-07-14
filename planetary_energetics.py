import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import copy
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.misc import derivative

class Layer(object):
    '''
    The layer base class defines the geometry of a spherical shell within
    a planet.
    '''

    def __init__( self, inner_radius, outer_radius, params={}):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.thickness = outer_radius-inner_radius

        assert self.thickness > 0.0

        self.inner_surface_area = 4.0 * np.pi * self.inner_radius**2.
        self.outer_surface_area = 4.0 * np.pi * self.outer_radius**2.

        self.volume = 4.0/3.0 * np.pi * ( self.outer_radius**3. - self.inner_radius**3.)
        self.params = params

    def set_boundary_temperatures(self,outer_temperature,inner_temperature): 
        '''
        All layers should be able to track the temperatures of the their outer and inner
        boundary.
        '''
        self.outer_temperature = outer_temperature
        self.inner_temperature = inner_temperature

    def ODE(y, t):
        raise NotImplementedError("Need to define an ODE")

    def lower_heat_flux_attempt (self):
        raise NotImplementedError("Need to define a heat flux function")

    def upper_heat_flux_attempt (self):
        raise NotImplementedError("Need to define a heat flux function")

class Planet(object):

    def __init__( self, layers):
        self.layers = layers
        self.Nlayers = len(layers)

        self.radius = self.layers[-1].outer_radius 
        self.volume = 4./3. * np.pi * self.radius**3

        self.core_layer = layers[0]
        self.mantle_layer = layers[1]

    def integrate( self, T_cmb_initial, T_mantle_initial, times):
        
        def ODE( temperatures, t ):
            dTmantle_dt = self.mantle_layer.mantle_energy_balance( t, temperatures[1], temperatures[0] )
            cmb_flux = self.mantle_layer.lower_boundary_flux( temperatures[1], temperatures[0] )
            dTcore_dt = self.core_layer.core_energy_balance(temperatures[0], cmb_flux )
            # print('\n\n time={0:.2f} Mya'.format(t/(np.pi*1e7*1e6)))
            return np.array([dTcore_dt, dTmantle_dt])

        solution = integrate.odeint( ODE, np.array([T_cmb_initial, T_mantle_initial]), times)
        return times, solution

    def draw(self):

        c = ['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc', \
                '#e5d8bd','#fddaec','#f2f2f2']
        fig = plt.figure()
        axes = fig.add_subplot(111)

        wedges = []
        for i,layer in enumerate(self.layers):
           wedges.append( patches.Wedge( (0.0,0.0), layer.outer_radius, 70.0, 110.0,\
                   width=layer.thickness, color=c[i]) )
        p = PatchCollection( wedges, match_original = True )
        axes.add_collection( p )
        r = max( [l.outer_radius for l in self.layers ] ) * 1.1

        axes.set_ylim( 0, r)
        axes.set_xlim( -r/2.0 , r/2.0 )
        plt.axis('off')
        plt.show()

class CoreLayer(Layer):
    def __init__(self,inner_radius,outer_radius, params={}):
        Layer.__init__(self,inner_radius,outer_radius, params)
        self.light_alloy = self.params.core.x_0

    def set_light_alloy_concentration(self):
        '''
        Equation (7) from Stevenson 1983
        '''
        pc = self.params.core
        R_c = self.inner_radius
        R_i = self.outer_radius
        self.light_alloy = pc.x_0*(R_c**3)/(R_c**3-R_i**3)
        return self.light_alloy

    def set_inner_core_radius(self, R_i):
        self.inner_radius = R_i
        return self.inner_radius

    ### We could code the integrals here. 
    def core_mantle_boundary_temp(self):
        return  self.T_average / self.mu

    def stevenson_liquidus(self, P):
        '''
        Equation (3) from Stevenson 1983
        
        Calculates the liquidus temp for a given pressure in the core P
        '''
        x  = self.light_alloy
        pc = self.params.core
        return pc.T_m0 * (1. - pc.alpha * x) * (1. + pc.T_m1 * P + pc.T_m2 * P**2.)
    
    def stevenson_adiabat(self, P, T_cmb):
        '''
        Equation (4) from Stevenson 1983

        Calculates adiabatic temperature for a given pressure within the core P, given the temperature at the CMB T_cmb
        '''
        pc = self.params.core
        return T_cmb * (1. + pc.T_a1*P + pc.T_a2*P**2.) / (1. + pc.T_a1*pc.P_cm + pc.T_a2*pc.P_cm**2.)
    
    def calculate_pressure_io_boundary(self, T_cmb):
        pc = self.params.core
        opt_function = lambda P: (self.stevenson_adiabat(P, T_cmb)-self.stevenson_liquidus(P))
        if self.stevenson_liquidus(pc.P_c) <= self.stevenson_adiabat(pc.P_c,T_cmb):
            P_io = pc.P_c
        elif self.stevenson_liquidus(pc.P_cm) >= self.stevenson_adiabat(pc.P_cm,T_cmb):
            P_io = pc.P_cm
        else:
            P_io = opt.brentq(opt_function, pc.P_c, pc.P_cm)
        return P_io

    def inner_core_radius(self, T_cmb): 
        '''
        Equation 5 from Stevenson et al 1983
        '''
        pc = self.params.core
        R_c  = self.outer_radius
        P_io = self.calculate_pressure_io_boundary( T_cmb )
        R_i  = max(0.,np.sqrt(2.*(pc.P_c - P_io)*R_c/(pc.rho*self.params.g)))
        return R_i

    def core_energy_balance(self, T_cmb, core_flux):
        pc = self.params.core
        core_surface_area = self.outer_surface_area
        inner_core_surface_area = np.power(self.inner_core_radius(T_cmb), 2.0) * 4. * np.pi
        dRi_dTcmb = 0.
        try:
            dRi_dTcmb = derivative( self.inner_core_radius, T_cmb, dx=1.0)
        except ValueError:
            pass
        thermal_energy_change = pc.rho*pc.C*self.volume*pc.mu
        # latent_heat = -pc.L_Eg * pc.rho * inner_core_surface_area * dRi_dTcmb
        # dTdt = -core_flux * core_surface_area / (thermal_energy_change-latent_heat)
        dTdt = -core_flux * core_surface_area / (thermal_energy_change)
        return dTdt

    def ODE( self, T_cmb_initial, cmb_flux ):
        dTdt = lambda x, t : self.core_energy_balance( x, cmb_flux )
        return dTdt

class MantleLayer(Layer):
    def __init__(self,inner_radius,outer_radius, params={}):
        Layer.__init__(self,inner_radius,outer_radius,params)

    def average_mantle_temp(self, T_upper_mantle):
        pm = self.params.mantle
        return  T_upper_mantle * pm.mu

    def kinematic_viscosity(self, T_upper_mantle):
        pm = self.params.mantle
        return pm.nu_0*np.exp(pm.A/T_upper_mantle)
    
    def heat_production(self, time):
        '''
        Equation (2) from Stevenson et al 1983
        '''
        pm = self.params.mantle
        return pm.Q_0*np.exp(-pm.lam*time)

    def lower_mantle_temperature(self, T_upper_mantle):
        '''
        Adiabatic Temperature Increase from the temperature at the base of upper mantle boundary layer to
        the top of the lower boundary layer assuming negligable boundary layer thickness.
        '''
        pm = self.params.mantle
        return T_upper_mantle*( 1.0 + pm.alpha*pm.g*self.thickness/pm.C)
    
    def mantle_rayleigh_number(self, T_upper_mantle, T_cmb):
        '''
        Equation (19) Stevenson et al 1983
        '''
        pm = self.params.mantle
        nu = self.kinematic_viscosity(T_upper_mantle)
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        upper_boundary_delta_T = T_upper_mantle - self.params.T_s
        lower_boundary_delta_T = T_cmb - T_lower_mantle
        # print("udT={0:.1f} K, ldT={1:.1f} K, nu={2:.2e} m^2/s".format(upper_boundary_delta_T, lower_boundary_delta_T, nu))
        assert upper_boundary_delta_T > 0.0
        assert lower_boundary_delta_T > 0.0
        delta_T_effective = upper_boundary_delta_T + lower_boundary_delta_T
        return pm.g*pm.alpha*( delta_T_effective)*np.power(self.thickness,3.)/(nu*pm.K)
    
    def boundary_layer_thickness(self, Ra):
        '''
        Equation (18) Stevenson et al 1983
        '''
        pm = self.params.mantle
        return self.thickness*np.power(pm.Ra_crit/Ra, pm.beta)

    def upper_boundary_layer_thickness(self, T_upper_mantle, T_cmb):
        '''
        Use Equations (18,19) from Stevenson et al 1983 
        '''
        Ra = self.mantle_rayleigh_number(T_upper_mantle, T_cmb)
        # print("Ra={0:.2e}".format(Ra))
        return self.boundary_layer_thickness(Ra)
    
    def lower_boundary_layer_thickness(self, T_upper_mantle, T_cmb):
        '''
        Equations (20,21) Stevenson et al 1983
        '''
        pm = self.params.mantle
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        delta_T_lower_boundary_layer = T_cmb - T_lower_mantle
        average_boundary_layer_temp = T_lower_mantle + delta_T_lower_boundary_layer/2.
        nu_crit = self.kinematic_viscosity(T_upper_mantle)
        # print('T_cmb={0:.1f} K, T_lm={1:.1f} K, T_um={2:.1f} K, T_lbl={3:.1f} K'.format(T_cmb, T_lower_mantle, T_upper_mantle, average_boundary_layer_temp))
        # import ipdb; ipdb.set_trace()
        assert delta_T_lower_boundary_layer > 0.0, "{0:.1f}, {1:.1f}, {2:.1f}".format(T_cmb, T_lower_mantle, T_upper_mantle)
        delta_c = np.power( pm.Ra_boundary_crit*nu_crit*pm.K/(pm.g*pm.alpha*delta_T_lower_boundary_layer), 0.333 )
        Ra_mantle = self.mantle_rayleigh_number(T_upper_mantle, T_cmb)
        delta_c_normal = self.boundary_layer_thickness(Ra_mantle)
        # print('LBLT normal = {0:.1f} km, LBLT inc. visc={1:.1f} km'.format(delta_c_normal/1e3, delta_c/1e3))
        return np.minimum(delta_c,  delta_c_normal)
        # return self.boundary_layer_thickness(Ra_mantle)

    def upper_boundary_flux(self, T_upper_mantle, T_cmb):
        '''
        Equation (17) from Stevenson et al 1983

        :param T_upper_mantle:
        :param T_cmb:
        :return:
        '''
        pm = self.params.mantle
        delta_T = T_upper_mantle - self.params.T_s
        upper_boundary_layer_thickness = self.upper_boundary_layer_thickness(T_upper_mantle, T_cmb)
        # print("uBLt = {0:.1f} km".format(upper_boundary_layer_thickness/1e3))
        return pm.k*delta_T/upper_boundary_layer_thickness

    def lower_boundary_flux(self, T_upper_mantle, T_cmb):
        '''
        Equation (17) from Stevenson et al 1983

        :param T_upper_mantle:
        :param T_cmb:
        :return:
        '''
        pm = self.params.mantle
        delta_T = T_cmb - self.lower_mantle_temperature(T_upper_mantle)
        lower_boundary_layer_thickness = self.lower_boundary_layer_thickness(T_upper_mantle, T_cmb)
        # print("LBLt = {0:.1f} km".format(lower_boundary_layer_thickness/1e3))
        return pm.k*delta_T/lower_boundary_layer_thickness

    def mantle_energy_balance(self, time, T_upper_mantle, T_cmb):
        pm = self.params.mantle
        mantle_surface_area = self.outer_surface_area
        core_surface_area   = self.inner_surface_area

        effective_heat_capacity = pm.rho*pm.C*pm.mu*self.volume
        internal_heat_energy = self.heat_production(time)*self.volume
        cmb_flux = self.lower_boundary_flux(T_upper_mantle, T_cmb)
        surface_flux = self.upper_boundary_flux(T_upper_mantle, T_cmb)
        net_flux_out = mantle_surface_area*surface_flux - core_surface_area*cmb_flux
        # net_flux_out = mantle_surface_area*surface_flux
        # print('CMB flux={0:.1f} TW, Surf flux={1:.1f} TW, net flux out={2:.1f} TW'.format(cmb_flux*core_surface_area/1e12, mantle_surface_area*surface_flux/1e12, net_flux_out/1e12))
        dTdt = (internal_heat_energy - net_flux_out)/effective_heat_capacity
        # print('heat={0:.1f}TW, effective heat capacity = {1:.1f}'.format(internal_heat_energy/1e12, effective_heat_capacity/1e24))
        return dTdt

    def ODE( self, T_u_initial, T_cmb ):
        dTdt = lambda x, t : self.mantle_energy_balance( t, x, T_cmb )
        return dTdt

class Parameters(object):
    def __init__(self, source):
        self.source = source
        pass

Stevenson = Parameters('Stevenson 1983')
Stevenson.R_p0 = 6371e3 # - [m] from Stevenson Table II
Stevenson.R_c0 = 3485e3 # - [m] from Stevenson pg. 474
Stevenson.g = 10. # - [m/s^2] from Stevenson Table II
Stevenson.T_s = 293. # - [K] from Stevenson Table II

Stevenson.mantle = Parameters('Stevenson 1983, for mantle')
Stevenson.mantle.mu = 1.3 # - [] from Stevenson pg. 473 and Table II
Stevenson.mantle.alpha = 2e-5 # - [/K] from Stevenson Table I
Stevenson.mantle.k = 4.0 # - [W/m-K] from Stevenson Table I
Stevenson.mantle.K = 1e-6 # - [m^2/s] from Stevenson Table I
Stevenson.mantle.rhoC = 4e6 # - [J/m^3-K] from Stevenson Table I
Stevenson.mantle.rho = 5000. # - [kg/m^3] -- guess as Stevenson never explicitly states his assumption for rho or C
Stevenson.mantle.C = Stevenson.mantle.rhoC/Stevenson.mantle.rho # - [J/K-kg]
Stevenson.mantle.Q_0 = 0.
# Stevenson.mantle.Q_0 = 1.7e-7 # - [W/m^3] from Stevenson Table I
Stevenson.mantle.lam = 1.38e-17 # - [1/s] from Stevenson Table I
Stevenson.mantle.A = 5.2e4 # - [K] from Stevenson Table I
Stevenson.mantle.nu_0 = 4.0e3 # - [m^2/s] from Stevenson Table I
Stevenson.mantle.Ra_crit = 5e2 # - [] from Stevenson Table I
Stevenson.mantle.beta = 0.3 # - [] from Stevenson Table I
Stevenson.mantle.g = Stevenson.g # - [m/s^2] from Stevenson Table II
Stevenson.mantle.Ra_boundary_crit = 2e3 # empirical parameter

Stevenson.core = Parameters('Stevenson 1983, for core')
Stevenson.core.rho = 13000. # - [kg/m^3] from Stevenson pg. 474
Stevenson.core.alpha = 2e-5 # - [/K] from Stevenson Table I
Stevenson.core.rhoC = Stevenson.mantle.rhoC # - [J/m^3-K] from Stevenson Table I
Stevenson.core.C = Stevenson.core.rhoC/Stevenson.core.rho
Stevenson.core.x_0 = 0.1 # - [wt% S] from Stevenson pg. 474
Stevenson.core.P_c = 360e9 # - [Pa] from Stevenson pg. 474
Stevenson.core.P_cm = 140e9 # - [Pa] from Stevenson pg. 474
Stevenson.core.mu = 1.2 # - [] from Stevenson pg. 473 and Table II
Stevenson.core.T_m1 = 6.14e-12 # - [K/Pa] from Stevenson Table II
Stevenson.core.T_m2 = -4.5e-24 # - [K/Pa^2] from Stevenson Table II
Stevenson.core.T_a1 = 3.96e-12 # - [K/Pa] from Stevenson Table II
Stevenson.core.T_a2 = -3.3e-24 # - [K/Pa^2] from Stevenson Table II

Stevenson_E1 = copy.deepcopy(Stevenson)
Stevenson_E1.core.L_Eg = 1e6 # - [J/kg] from Stevenson Table III
Stevenson_E1.core.T_m0 = 1950. # - [K] from Stevenson Table III

Stevenson_E2 = copy.deepcopy(Stevenson)
Stevenson_E2.core.L_Eg = 2e6 # - [J/kg] from Stevenson Table III
Stevenson_E2.core.T_m0 = 1980. # - [K] from Stevenson Table III

#%%
Earth = Planet( [ CoreLayer( 0.0, Stevenson_E1.R_c0, params=Stevenson_E1) , MantleLayer( Stevenson_E1.R_c0, Stevenson_E1.R_p0, params=Stevenson_E1) ] )
#%%
T_cmb_initial = 5500.
T_mantle_initial = 3100.
Earth_age_yr = 4350e6*365.25*24.*3600.
times = np.linspace(0., Earth_age_yr, 1000)

t, y = Earth.integrate(T_cmb_initial, T_mantle_initial, times)
plt.plot( t, y[:,0])
plt.plot( t, y[:,1])
plt.show()
