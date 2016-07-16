import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import copy
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.misc import derivative
from input_parameters import *

class Planet(object):

    def __init__( self, layers):
        self.layers = layers
        self.Nlayers = len(layers)

        self.radius = self.layers[-1].outer_radius 
        self.volume = 4./3. * np.pi * self.radius**3

        self.core_layer = layers[0]
        self.magma_ocean_layer = layers[1]
        self.mantle_layer = layers[2]
        self.core_layer.planet = self
        self.magma_ocean_layer.planet = self
        self.mantle_layer.planet = self
        print("R_p={0:.1f} km, R_mo={1:.1f} km, R_c={2:.1f} km".format(self.mantle_layer.outer_radius, self.magma_ocean_layer.outer_radius, self.core_layer.outer_radius))

    def integrate( self, T_cmb_initial, T_magma_ocean_initial, T_mantle_initial, times, verbose=False):
        self.D_mo = []
        self.t_all = []
        self.T_umo = []
        def ODE( values, t ):
            P_mo = self.magma_ocean_layer.calculate_pressure_magma_ocean_top()
            T_liq = self.magma_ocean_layer.calculate_liquidus_temp(P_mo)
            T_sol = self.magma_ocean_layer.calculate_solidus_temp(P_mo)
            Dlbl_mo = self.magma_ocean_layer.lower_boundary_layer_thickness(values[1], values[0])
            T_umo = self.magma_ocean_layer.upper_temperature(values[1])
            if verbose:
                print("\ntime={0:.4f} Myr".format(t/(365.25*24.*3600.*1e6)))
                print("T_cmb={0:.1f} K".format(values[0]))
                print("T_lower_mo={0:.1f} K, D_lbl_mo={1:.3f} m".format(values[1], Dlbl_mo))
                print("T_upper_mo = {0:.1f} K, T_liq = {1:.1f} K, T_sol={2:.1f} K, D = {3:.1f} km".format(T_umo, T_liq, T_sol, self.magma_ocean_layer.thickness/1e3))
                print("T_lower_mantle={0:.1f} K".format(self.mantle_layer.lower_mantle_temperature(values[2])))
                print("T_upper_mantle={0:.1f} K".format(values[2]))

            dTmantle_dt = self.mantle_layer.mantle_energy_balance( values[2], values[1], t )
            mantle_bottom_flux = self.mantle_layer.lower_boundary_flux( values[2], values[1] )
            dTmagmaocean_dt = self.magma_ocean_layer.magma_ocean_energy_balance(values[1], values[0], mantle_bottom_flux, t)
            magma_ocean_bottom_flux = self.magma_ocean_layer.lower_boundary_flux(values[1], values[0], mantle_bottom_flux)
            dTcore_dt = self.core_layer.core_energy_balance(values[0], magma_ocean_bottom_flux)
            self.magma_ocean_layer.update_boundary_location(values[1])
            if verbose:
                print("mantle flux={0:.3f} W/m^2".format(mantle_bottom_flux))
                print("magma ocean flux={0:.3f} W/m^2".format(magma_ocean_bottom_flux))
            self.t_all.append(t)
            self.D_mo.append(self.magma_ocean_layer.thickness)
            self.T_umo.append(T_umo)
            return np.array([dTcore_dt, dTmagmaocean_dt, dTmantle_dt])

        solution = integrate.odeint( ODE, np.array([T_cmb_initial, T_magma_ocean_initial, T_mantle_initial]), times)

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

class Layer(object):
    '''
    The layer base class defines the geometry of a spherical shell within
    a planet.
    '''

    def __init__( self, inner_radius, outer_radius, params={}):
        self.set_boundaries(inner_radius, outer_radius)
        self.params = params

    def set_boundaries(self, inner_radius, outer_radius):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.thickness = outer_radius-inner_radius

        assert self.thickness >= 0.0, "Ri={0:.1f} km, Ro={1:.1f} km".format(self.inner_radius/1e3, self.outer_radius/1e3)

        self.inner_surface_area = 4.0 * np.pi * self.inner_radius**2.
        self.outer_surface_area = 4.0 * np.pi * self.outer_radius**2.

        self.volume = 4.0/3.0 * np.pi * ( self.outer_radius**3. - self.inner_radius**3.)

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

class CoreLayer(Layer):
    def __init__(self, inner_radius, outer_radius, params={}):
        Layer.__init__(self, inner_radius, outer_radius, params)
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
        if self.params.source == 'Stevenson_1983' :
            inner_core_surface_area = np.power(self.inner_core_radius(T_cmb), 2.0) * 4. * np.pi
            dRi_dTcmb = 0.
            try:
                dRi_dTcmb = derivative( self.inner_core_radius, T_cmb, dx=1.0)
            except ValueError:
                pass
        elif self.params.source == 'Driscoll_2014' :
            dRi_dTcmb = 0.
            inner_core_surface_area = 0.
            # Eqn 29 & 30 from the Driscoll_2014 paper. Note that Eqn 32 in the paper has an error and the form in the code is correct
            sqrt_term_num = (pc.Dn/self.params.R_c0)**2.*np.log(pc.TFe/T_cmb) - 1.
            sqrt_term_den = 2.*(1. - 1./(3.*pc.gamma_c))*(pc.Dn/pc.DFe)**2. - 1.
            if sqrt_term_num/sqrt_term_den > 0 :
                R_ic = self.params.R_c0*np.sqrt(sqrt_term_num/sqrt_term_den)
                print('\n\n R_ic={0:.3f}'.format(R_ic/1e3))
                inner_core_surface_area = np.power(R_ic, 2.0) * 4. * np.pi
                dRi_dTcmb = -1.*((self.params.R_c0/2./T_cmb)*(pc.Dn/self.params.R_c0)**2.)/(sqrt_term_num*sqrt_term_den)
        else :
            raise ValueError('parameter class is not recognized')
        # print('\n\n ratio={0:.1f}'.format(inner_core_surface_area/core_surface_area))
        thermal_energy_change = pc.rho*pc.C*self.volume*pc.mu
        latent_heat = -pc.L_Eg * pc.rho * inner_core_surface_area * dRi_dTcmb
        dTdt = -core_flux * core_surface_area / (thermal_energy_change-latent_heat)
        dTdt = -core_flux * core_surface_area / (thermal_energy_change)
        return dTdt

    def ODE( self, T_cmb_initial, cmb_flux ):
        dTdt = lambda x, t : self.core_energy_balance( x, cmb_flux )
        return dTdt

class MagmaOceanLayer(Layer):
    def __init__(self,inner_radius,outer_radius, params={}):
        Layer.__init__(self,inner_radius,outer_radius, params)

    def lower_temperature(self, T_magma_ocean):
        '''
        Adiabatic Temperature Increase from the temperature at the base of upper mantle boundary layer to
        the top of the lower boundary layer assuming negligable boundary layer thickness.
        '''
        po = self.params.magma_ocean
        return T_magma_ocean*( 1.0 + po.alpha*po.g*self.thickness/po.C)

    def upper_temperature(self, T_magma_ocean):
        '''
        Adiabatic Temperature Increase from the temperature at the base of upper mantle boundary layer to
        the top of the lower boundary layer assuming negligable boundary layer thickness.
        '''
        po = self.params.magma_ocean
        return T_magma_ocean*( 1.0 - po.alpha*po.g*self.thickness/po.C)

    def rayleigh_number(self, T_upper, T_lower):
        '''

        :param T_mantle_bottom:
        :param T_lower:
        :return:
        '''
        po = self.params.magma_ocean
        T_avg = (T_upper + T_lower)/2
        delta_T = T_lower-T_upper
        assert delta_T >= 0., 'dT={0:.1f} K'.format(delta_T)
        return po.g*po.alpha*delta_T*np.power(self.thickness,3.)/(po.nu*po.K)
        pass

    def calculate_dTdt_adiabat(self, T_magma_ocean):
        po = self.params.magma_ocean
        P_mo = self.calculate_pressure_magma_ocean_top()
        T_liq = self.calculate_liquidus_temp(P_mo)
        T_sol = self.calculate_solidus_temp(P_mo)
        R = self.outer_radius
        if T_magma_ocean < T_liq and T_magma_ocean > T_sol:
            dTdt_a = (-po.alpha*po.g/po.C)*self.volume/((T_sol-T_liq)*4*np.pi*R**2)
        else:
            dTdt_a = 0.
        return dTdt_a

    def calculate_solidus_temp(self, P):
        po = self.params.magma_ocean
        return po.c1_sol*(P/po.c2_sol + 1)**(1/po.c3_sol)

    def calculate_liquidus_temp(self, P):
        po = self.params.magma_ocean
        return po.c1_liq*(P/po.c2_liq + 1)**(1/po.c3_liq)

    def calculate_pressure_magma_ocean_top(self):
        '''
        Calculates the pressure at the top of the magma ocean using the pressure at the CMB and rho*g*thickness
        :return:
        '''
        po = self.params.magma_ocean
        return self.params.core.P_c - self.thickness*po.rho*po.g

    def calculate_solidification_fraction(self, T_upper_magma_ocean):
        '''
        Calculates the fraction of the layer that solidifies assuming a uniform layer temperature and a linear phase diagram

        :param T:
        :return:
        '''
        P_mo = self.calculate_pressure_magma_ocean_top()
        T_sol = self.calculate_solidus_temp(P_mo)
        T_liq = self.calculate_liquidus_temp(P_mo)
        if T_upper_magma_ocean < T_sol:
            return 1.0
        elif T_upper_magma_ocean > T_liq:
            return 0.0
        else:
            return 1.-(T_upper_magma_ocean-T_sol)/(T_liq-T_sol)

    def calculate_thickness_change(self, T_magma_ocean):
        T_upper = self.upper_temperature(T_magma_ocean)
        sol_frac = self.calculate_solidification_fraction(T_upper)
        new_volume = self.volume*(1-sol_frac)
        new_thickness = (3*new_volume/(4*np.pi) + self.inner_radius**3.)**(1./3.) - self.inner_radius
        return new_thickness-self.thickness

    def update_boundary_location(self, T_magma_ocean):
        new_thickness = max(self.calculate_thickness_change(T_magma_ocean)+self.thickness, 0.0)
        self.set_boundaries(self.inner_radius, self.inner_radius+new_thickness)
        self.planet.mantle_layer.set_boundaries(self.outer_radius, self.planet.mantle_layer.outer_radius)

    def heat_production(self, time):
        '''
        Equation (2) from Stevenson et al 1983
        '''
        po = self.params.magma_ocean
        return po.Q_0*np.exp(-po.lam*time)

    def calculate_latent_heat(self, T_upper_magma_ocean):
        po = self.params.magma_ocean
        P_mo = self.calculate_pressure_magma_ocean_top()
        T_sol = self.calculate_liquidus_temp(P_mo)
        T_liq = self.calculate_solidus_temp(P_mo)
        if T_upper_magma_ocean < T_liq and T_upper_magma_ocean > T_sol:
            latent_heat = po.L_Eg*po.rho*self.volume/(T_liq-T_sol)
        else:
            latent_heat = 0.
        return latent_heat

    def boundary_layer_thickness(self, Ra):
        '''
        Equation (18) Stevenson et al 1983
        '''
        po = self.params.magma_ocean
        if Ra > 0.:
            return self.thickness*np.power(po.Ra_crit/Ra, po.beta)
        else:
            return self.thickness

    def lower_boundary_layer_thickness(self, T_magma_ocean, T_cmb):
        '''
        Equations (20,21) Stevenson et al 1983
        '''
        po = self.params.magma_ocean
        T_upper_magma_ocean = self.upper_temperature(T_magma_ocean)
        delta_T_lower_boundary_layer = T_cmb - T_magma_ocean
        assert delta_T_lower_boundary_layer > 0.0, "dTlbl_mo={0:.1f} K, Tl_mo={1:.1f} K, Tcmb={2:.1f} K".format(delta_T_lower_boundary_layer, T_lower, T_cmb)
        Ra = self.rayleigh_number(T_upper_magma_ocean, T_magma_ocean)
        delta = self.boundary_layer_thickness(Ra)
        return delta

    def lower_boundary_flux(self, T_magma_ocean, T_cmb, mantle_bottom_flux):
        '''
        Equation (17) from Stevenson et al 1983

        :param T_upper_mantle:
        :param T_mantle_bottom:
        :return:
        '''
        po = self.params.magma_ocean
        lower_boundary_layer_thickness = self.lower_boundary_layer_thickness(T_magma_ocean, T_cmb)
        if self.thickness > 100.:
            delta_T = T_cmb - T_magma_ocean
            assert delta_T > 0., "dT={0:.1f} K".format(delta_T)
            return po.k*delta_T/lower_boundary_layer_thickness
        else:
            return mantle_bottom_flux

    def magma_ocean_energy_balance(self, T_magma_ocean, T_cmb, mantle_bottom_flux, time):
        po = self.params.magma_ocean
        if self.thickness > 100.:
            T_upper = self.upper_temperature(T_magma_ocean)
            latent_heat = self.calculate_latent_heat(T_upper)
            effective_heat_capacity = po.rho*po.C*po.mu*self.volume
            internal_heat_energy = self.heat_production(time)*self.volume
            cmb_flux = self.lower_boundary_flux(T_magma_ocean, T_cmb, mantle_bottom_flux)
            net_flux_out = self.outer_surface_area*mantle_bottom_flux - self.inner_surface_area*cmb_flux
            dTdt = (internal_heat_energy - net_flux_out)/(effective_heat_capacity - latent_heat)
            # print("dTdt={0:.3e} K, dTdt_a={1:.3e} K".format(dTdt, ))
        else:
            dTdt = self.planet.core_layer.core_energy_balance(T_cmb, mantle_bottom_flux)
        return dTdt

    def ODE( self, D_magma_ocean_initial, T_mag):
        dDdt = lambda x, t : self.magma_ocean_energy_balance( x, )
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
    
    def mantle_rayleigh_number(self, T_upper_mantle, T_mantle_bottom):
        '''
        Equation (19) Stevenson et al 1983
        '''
        pm = self.params.mantle
        nu = self.kinematic_viscosity(T_upper_mantle)
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        upper_boundary_delta_T = T_upper_mantle - self.params.T_s
        lower_boundary_delta_T = T_mantle_bottom - T_lower_mantle
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

    def upper_boundary_layer_thickness(self, T_upper_mantle, T_mantle_bottom):
        '''
        Use Equations (18,19) from Stevenson et al 1983 
        '''
        Ra = self.mantle_rayleigh_number(T_upper_mantle, T_mantle_bottom)
        return self.boundary_layer_thickness(Ra)
    
    def lower_boundary_layer_thickness(self, T_upper_mantle, T_mantle_bottom):
        '''
        Equations (20,21) Stevenson et al 1983
        '''
        pm = self.params.mantle
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        delta_T_lower_boundary_layer = T_mantle_bottom - T_lower_mantle
        average_boundary_layer_temp = T_lower_mantle + delta_T_lower_boundary_layer/2.
        nu_crit = self.kinematic_viscosity(T_upper_mantle)
        # print('T_mantle_bottom={0:.1f} K, T_lm={1:.1f} K, T_um={2:.1f} K, T_lbl={3:.1f} K'.format(T_mantle_bottom, T_lower_mantle, T_upper_mantle, average_boundary_layer_temp))
        # import ipdb; ipdb.set_trace()
        assert delta_T_lower_boundary_layer > 0.0, "dTlbl={3:.1f} K, T_mb={0:.1f} K, T_lm={1:.1f} K, T_um={2:.1f} K".format(T_mantle_bottom, T_lower_mantle, T_upper_mantle, delta_T_lower_boundary_layer)
        delta_c = np.power( pm.Ra_boundary_crit*nu_crit*pm.K/(pm.g*pm.alpha*delta_T_lower_boundary_layer), 0.333 )
        Ra_mantle = self.mantle_rayleigh_number(T_upper_mantle, T_mantle_bottom)
        delta_c_normal = self.boundary_layer_thickness(Ra_mantle)
        # print('LBLT normal = {0:.1f} km, LBLT inc. visc={1:.1f} km'.format(delta_c_normal/1e3, delta_c/1e3))
        return np.minimum(delta_c,  delta_c_normal)
        # return self.boundary_layer_thickness(Ra_mantle)

    def upper_boundary_flux(self, T_upper_mantle, T_mantle_bottom):
        '''
        Equation (17) from Stevenson et al 1983

        :param T_upper_mantle:
        :param T_mantle_bottom:
        :return:
        '''
        pm = self.params.mantle
        delta_T = T_upper_mantle - self.params.T_s
        upper_boundary_layer_thickness = self.upper_boundary_layer_thickness(T_upper_mantle, T_mantle_bottom)
        # print("uBLt = {0:.1f} km".format(upper_boundary_layer_thickness/1e3))
        return pm.k*delta_T/upper_boundary_layer_thickness

    def lower_boundary_flux(self, T_upper_mantle, T_mantle_bottom):
        '''
        Equation (17) from Stevenson et al 1983

        :param T_upper_mantle:
        :param T_mantle_bottom:
        :return:
        '''
        pm = self.params.mantle
        delta_T = T_mantle_bottom - self.lower_mantle_temperature(T_upper_mantle)
        lower_boundary_layer_thickness = self.lower_boundary_layer_thickness(T_upper_mantle, T_mantle_bottom)
        # print("LBLt = {0:.1f} km".format(lower_boundary_layer_thickness/1e3))
        return pm.k*delta_T/lower_boundary_layer_thickness

    def mantle_energy_balance(self, T_upper_mantle, T_mantle_bottom, time):
        pm = self.params.mantle
        mantle_surface_area = self.outer_surface_area
        core_surface_area   = self.inner_surface_area

        effective_heat_capacity = pm.rho*pm.C*pm.mu*self.volume
        internal_heat_energy = self.heat_production(time)*self.volume
        cmb_flux = self.lower_boundary_flux(T_upper_mantle, T_mantle_bottom)
        surface_flux = self.upper_boundary_flux(T_upper_mantle, T_mantle_bottom)
        net_flux_out = mantle_surface_area*surface_flux - core_surface_area*cmb_flux
        # net_flux_out = mantle_surface_area*surface_flux
        # print('CMB flux={0:.1f} TW, Surf flux={1:.1f} TW, net flux out={2:.1f} TW'.format(cmb_flux*core_surface_area/1e12, mantle_surface_area*surface_flux/1e12, net_flux_out/1e12))
        dTdt = (internal_heat_energy - net_flux_out)/effective_heat_capacity
        # print('heat={0:.1f}TW, effective heat capacity = {1:.1f}'.format(internal_heat_energy/1e12, effective_heat_capacity/1e24))
        return dTdt

    def ODE( self, T_u_initial, T_mantle_bottom ):
        dTdt = lambda x, t : self.mantle_energy_balance( t, x, T_mantle_bottom )
        return dTdt

Stevenson_E1 = Stevenson_1983(case=1)
Stevenson_E2 = Stevenson_1983(case=2)
Driscoll = Driscoll_2014()

param2layer = Driscoll

# Earth = Planet( [ CoreLayer( 0.0, param2layer.R_c0, params=param2layer) , MantleLayer( param2layer.R_c0, param2layer.R_p0, params=param2layer) ] )

Andrault_f_perioditic = Andrault_2011_Stevenson(composition="f_perioditic", Stevenson_case=1)
Andrault_a_chondritic = Andrault_2011_Stevenson(composition="a_chondritic", Stevenson_case=1)

# param3layer = Andrault_f_perioditic
param3layer = Andrault_a_chondritic
Earth = Planet( [ CoreLayer( 0.0, param3layer.R_c0, params=param3layer) ,
                  MagmaOceanLayer(param3layer.R_c0, param3layer.R_mo0, params=param3layer),
                  MantleLayer(param3layer.R_mo0, param3layer.R_p0, params=param3layer) ] )
#%%
# T_cmb_initial = 8200. # K
# T_magma_ocean_initial = 7245. # K
# T_mantle_initial = 4200. # K
T_cmb_initial = 9500. # K
T_magma_ocean_initial = 8400. # K
T_mantle_initial = 5200. # K
end_time_Mya = 4568 # Mya
# end_time_Mya = 14 # Mya
end_time = end_time_Mya*1e6*const_yr_to_sec# s

Nt = 40000
times = np.linspace(0., end_time, Nt)
t, y = Earth.integrate(T_cmb_initial, T_magma_ocean_initial, T_mantle_initial, times, verbose=False)
t_plt = t/(365.25*24.*3600.*1e6)
t_pltind = Nt

def filter_ODE(t,data):
    tn = []
    datan = []
    N = len(t)
    last = t[N-1]
    for ind in range(len(t)):
        if t[N-ind-1] < last:
            tn.append(t[N-ind-1])
            datan.append(data[N-ind-1])
            last = t[N-ind-1]
    tout = np.array(tn)[::-1]
    dataout = np.array(datan)[::-1]
    return tout, dataout

t_all = np.array(Earth.t_all)/(365.25*24.*3600.*1e6)
D = np.array(Earth.D_mo)
T_umo = np.array(Earth.T_umo)
tD, D = filter_ODE(t_all, D)
tD, T_umo = filter_ODE(t_all, T_umo)

plt.plot( t_plt[:t_pltind], y[:t_pltind,0])
plt.plot( t_plt[:t_pltind], y[:t_pltind,1])
plt.plot( tD, T_umo)
plt.plot( t_plt[:t_pltind], y[:t_pltind,2])
plt.plot( tD, D/1e3)
plt.title("Thermal Evolution of Earth")
plt.ylabel(r"Temperature (K) or thickness (km)")
plt.xlabel(r"Time (Myr)")
plt.legend(["T - Core Mantle Boundary", "T - Lower Magma Ocean", "T - Upper Magma Ocean", "T - Upper Mantle", "Thickness of Magma Ocean"], loc=0)
plt.savefig("thermal_evolution_T_cmb{0:.0f}K_t{1:.0f}Myr.png".format(T_cmb_initial, end_time_Mya))
