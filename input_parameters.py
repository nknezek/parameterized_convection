import numpy as np

const_yr_to_sec = 3.154e7 # seconds

class Parameters(object):
    def __init__(self, source):
        self.source = source
        pass

class Stevenson_1983(Parameters) :
    def __init__(self, case):
        self.source = 'Stevenson_1983'
        self.R_p0 = 6371e3 # - [m] from self Table II
        self.R_c0 = 3485e3 # - [m] from self pg. 474
        self.g = 10. # - [m/s^2] from self Table II
        self.T_s = 293. # - [K] from self Table II

        self.mantle = Parameters('Stevenson_1983, for mantle')
        self.mantle.mu = 1.3 # - [] from self pg. 473 and Table II
        self.mantle.alpha = 2e-5 # - [/K] from self Table I
        self.mantle.k = 4.0 # - [W/m-K] from self Table I
        self.mantle.K = 1e-6 # - [m^2/s] from self Table I
        self.mantle.rhoC = 4e6 # - [J/m^3-K] from self Table I
        self.mantle.rho = 5000. # - [kg/m^3] -- guess as self never explicitly states his assumption for rho or C
        self.mantle.C = self.mantle.rhoC/self.mantle.rho # - [J/K-kg]
        self.mantle.Q_0 = 1.7e-7 # - [W/m^3] from self Table I
        self.mantle.lam = 1.38e-17 # - [1/s] from self Table I
        self.mantle.A = 5.2e4 # - [K] from self Table I
        self.mantle.nu_0 = 4.0e3 # - [m^2/s] from self Table I
        self.mantle.Ra_crit = 5e2 # - [] from self Table I
        self.mantle.beta = 0.3 # - [] from self Table I
        self.mantle.g = self.g # - [m/s^2] from self Table II
        self.mantle.Ra_boundary_crit = 2e3 # empirical parameter

        self.core = Parameters('Stevenson_1983, for core')
        self.core.rho = 13000. # - [kg/m^3] from self pg. 474
        self.core.alpha = 2e-5 # - [/K] from self Table I
        self.core.rhoC = self.mantle.rhoC # - [J/m^3-K] from self Table I
        self.core.C = self.core.rhoC/self.core.rho
        self.core.x_0 = 0.1 # - [wt% S] from self pg. 474
        self.core.P_c = 360e9 # - [Pa] from self pg. 474
        self.core.P_cm = 140e9 # - [Pa] from self pg. 474
        self.core.mu = 1.2 # - [] from self pg. 473 and Table II
        self.core.T_m1 = 6.14e-12 # - [K/Pa] from self Table II
        self.core.T_m2 = -4.5e-24 # - [K/Pa^2] from self Table II
        self.core.T_a1 = 3.96e-12 # - [K/Pa] from self Table II
        self.core.T_a2 = -3.3e-24 # - [K/Pa^2] from self Table II
        self.set_inner_core_L_Eg(case)

    def set_inner_core_L_Eg(self,case):
            if case ==1 :
                self.core.L_Eg = 1e6 # - [J/kg] from self Table III
                self.core.T_m0 = 1950. # - [K] from self Table III
            elif case ==2 :
                self.core.L_Eg = 2e6 # - [J/kg] from self Table III
                self.core.T_m0 = 1980. # - [K] from self Table III
            else:
                raise ValueError("case must be integer 1 for E1 or 2 for E2")

class Driscoll_2014(Parameters) :
    def __init__(self):
        self.source = 'Driscoll_2014'
        self.R_p0 = 6371e3 # - [m] from  Table III
        self.R_c0 = 3480e3 # - [m] from  Table III
        self.g = 10. # - [m/s^2] from  Table III, average g
        self.T_s = 290. # - [K] from pg. 44

        self.mantle = Parameters('Driscoll_2014, for mantle')
        self.mantle.mu = 1.3 # - [] from self pg. 473 and Table III
        self.mantle.alpha = 3e-5 # - [/K] from self Table III
        self.mantle.k = 4.2 # - [W/m-K] from self Table III , for upper mantle maybe be a factor of 2-3 larger in lower mantle
        self.mantle.K = 1e-6 # - [m^2/s] from self Table III
        self.mantle.rho = 4800. # - [kg/m^3] -- guess as self never explicitly states his assumption for rho or C
        self.mantle.C = 1265. # - [J/K-kg]
        self.mantle.rhoC = self.mantle.C*self.mantle.rho  # - [J/m^3-K] from self Table I
        self.mantle.Q_0 = 1.434e-8 # - [W/m^3] from self Table III
        self.mantle.lam = 1./2.94/const_yr_to_sec/1e9 # - [1/s] from self Table III
        self.mantle.A =  36081.7 # - [K] from self Table III (divide A/R_gas constant)
        self.mantle.nu_0 =  7e7 # - [m^2/s] from self Table III
        self.mantle.Ra_crit = 660. # - [] from self Table III
        self.mantle.beta = 1./3. # - [] from self Table III (allow a range from .2 to .3)
        self.mantle.g = self.g # - [m/s^2] from self Table III
        self.mantle.Ra_boundary_crit = 2e3 # empirical parameter
        self.mantle.vlm_vum = 10 # ratio of lower mantle to upper mantle viscisity
        self.mantle.lheat  = 320e3 # J/kg ->latent heat of mantle melting

        self.core = Parameters('Driscoll_2014, for core')
        self.core.rho = 11900. # - [kg/m^3] from self pg. 474
        self.core.alpha = 1e-5 # - [/K] from self Table III
        self.core.C = 840.  # - [J/K-kg]
        self.core.rhoC = self.core.C*self.core.rho # - [J/m^3-K] from self Table III
        self.core.x_0 = 0.0 # - [wt% S] - the solidus is for pure iron instead of a mixed alloy
        self.core.P_cm = self.mantle.rho*self.g*(self.R_p0 - self.R_c0) # - [Pa] from self pg. 474
        self.core.P_c =  self.mantle.rho*self.g*(self.R_p0 - self.R_c0) + self.core.rho*self.g*(self.R_c0) # - [Pa] from self pg. 474
        self.core.mu = 1.2  # - [] from self pg. 473 and Table II
        self.core.T_m1 = 6.14e-12 # - [K/Pa] from self Table II
        self.core.T_m2 = -4.5e-24 # - [K/Pa^2] from self Table II
        self.core.T_a1 = 3.96e-12 # - [K/Pa] from self Table II
        self.core.T_a2 = -3.3e-24 # - [K/Pa^2] from self Table II
        self.core.L_Eg = 3e5 + 750e3  # - [J/kg] from self Table III
        self.core.T_m0 = 1950. # - [K] from self Table III
        self.core.Dn = 6340e3#  adiabatic length scale,Pg.41
        self.core.DFe = 7000.e3 # constant length scale , Pg.41
        self.core.gamma_c = 1.3 # core Gruneisen parameter
        self.core.TFe = 5600. # kelvin, Pg 41
        self.core.lheat  = 750e3 # J/kg ->latent heat of mantle melting
        self.core.lam = 1./1.2/const_yr_to_sec/1e9 # - [1/s] from self Table III
        self.core.Q_0_up = 1.133e-9 # - [W/m^3] from Pg 41
        self.core.Q_0_low = 6.798e-9 # - [W/m^3] from Pg 41

class Andrault_2011_Stevenson(Stevenson_1983):
    def __init__(self, composition, Stevenson_case):
        Stevenson_1983.__init__(self, Stevenson_case)
        self.D_mo0 = 500e3 # - [km] initial thickness of magma ocean guess
        self.R_mo0 = self.R_c0 + self.D_mo0
        self.magma_ocean = Parameters('From Stevenson E1 and Andrault')
        self.magma_ocean.c1_sol = 2081.8 # - [K] from Moneteux 2016 (12) citing Andrault 2011
        self.magma_ocean.c2_sol = 101.69e9 # - [Pa] from Moneteux 2016 (12) citing Andrault 2011
        self.magma_ocean.c3_sol = 1.226 # - [] from Moneteux 2016 (12) citing Andrault 2011
        self.magma_ocean.alpha = 2e-5 # - [/K] from Stevenson Table I for mantle
        self.magma_ocean.k = 4.0 # - [W/m-K] from Stevenson Table I for mantle
        self.magma_ocean.K = 1e-6 # - [m^2/s] from Stevenson Table I for mantle
        self.magma_ocean.rhoC = 4e6 # - [J/m^3-K] from Stevenson Table I for mantle
        self.magma_ocean.rho = 5000. # - [kg/m^3] -- guess as Stevenson never explicitly states his assumption for rho or C
        self.magma_ocean.C = self.magma_ocean.rhoC/self.magma_ocean.rho # - [J/K-kg]
        self.magma_ocean.L_Eg = 3e5 # - [J/kg] guess
        self.magma_ocean.Q_0 = 1.7e-7 # - [W/m^3] from Stevenson Table I
        self.magma_ocean.lam = 1.38e-17 # - [1/s] from Stevenson Table I
        self.magma_ocean.g = self.g # - [m/s^2] from Stevenson Table II
        self.magma_ocean.nu = 1e-1 # - [m^2/s] -- estimate
        self.magma_ocean.mu = 1. # - [] -- ratio of average layer temperature to T_magma_ocean at top estimate
        self.magma_ocean.Ra_crit = 5e2 # - [] from Stevenson Table I
        self.magma_ocean.Ra_boundary_crit = 2e3 # empirical parameter
        self.magma_ocean.beta = 0.3 # - [] from Stevenson Table I
        self.set_composition(composition)

    def set_composition(self, composition):
        if composition == "f_perioditic":
            self.magma_ocean.c1_liq = 78.74 # - [K] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c2_liq = 4.054e6 # - [Pa] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c3_liq = 2.44 # - [] from Moneteux 2016 (13) citing Andrault 2011
        elif composition == "a_chondritic":
            self.magma_ocean.c1_liq = 2006.8 # - [K] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c2_liq = 34.65e9 # - [Pa] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c3_liq = 1.844 # - [] from Moneteux 2016 (13) citing Andrault 2011
        else:
            raise ValueError("Composition for Andrault_2011 must be f_perioditic or a_chondrtic")

class Andrault_2011_Driscoll(Driscoll_2014):
    def __init__(self, composition, Stevenson_case):
        Stevenson_1983.__init__(self, Stevenson_case)
        self.D_mo0 = 500e3 # - [km] initial thickness of magma ocean guess
        self.R_mo0 = self.R_c0 + self.D_mo0
        self.magma_ocean = Parameters('From Stevenson E1 and Andrault')
        self.magma_ocean.c1_sol = 2081.8 # - [K] from Moneteux 2016 (12) citing Andrault 2011
        self.magma_ocean.c2_sol = 101.69e9 # - [Pa] from Moneteux 2016 (12) citing Andrault 2011
        self.magma_ocean.c3_sol = 1.226 # - [] from Moneteux 2016 (12) citing Andrault 2011
        self.magma_ocean.alpha = 2e-5 # - [/K] from Stevenson Table I for mantle
        self.magma_ocean.k = 4.0 # - [W/m-K] from Stevenson Table I for mantle
        self.magma_ocean.K = 1e-6 # - [m^2/s] from Stevenson Table I for mantle
        self.magma_ocean.rhoC = 4e6 # - [J/m^3-K] from Stevenson Table I for mantle
        self.magma_ocean.rho = 5000. # - [kg/m^3] -- guess as Stevenson never explicitly states his assumption for rho or C
        self.magma_ocean.C = self.magma_ocean.rhoC/self.magma_ocean.rho # - [J/K-kg]
        self.magma_ocean.L_Eg = 3e5 # - [J/kg] guess
        self.magma_ocean.Q_0 = 1.7e-7 # - [W/m^3] from Stevenson Table I
        self.magma_ocean.lam = 1.38e-17 # - [1/s] from Stevenson Table I
        self.magma_ocean.g = self.g # - [m/s^2] from Stevenson Table II
        self.magma_ocean.nu = 1e-1 # - [m^2/s] -- estimate
        self.magma_ocean.mu = 1. # - [] -- ratio of average layer temperature to T_magma_ocean at top estimate
        self.magma_ocean.Ra_crit = 5e2 # - [] from Stevenson Table I
        self.magma_ocean.Ra_boundary_crit = 2e3 # empirical parameter
        self.magma_ocean.beta = 0.3 # - [] from Stevenson Table I
        self.set_composition(composition)

    def set_composition(self, composition):
        if composition == "f_perioditic":
            self.magma_ocean.c1_liq = 78.74 # - [K] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c2_liq = 4.054e6 # - [Pa] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c3_liq = 2.44 # - [] from Moneteux 2016 (13) citing Andrault 2011
        elif composition == "a_chondritic":
            self.magma_ocean.c1_liq = 2006.8 # - [K] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c2_liq = 34.65e9 # - [Pa] from Moneteux 2016 (13) citing Andrault 2011
            self.magma_ocean.c3_liq = 1.844 # - [] from Moneteux 2016 (13) citing Andrault 2011
        else:
            raise ValueError("Composition for Andrault_2011 must be f_perioditic or a_chondrtic")
