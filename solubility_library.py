"""
Created on Sun Jul 17 17:11:58 2016

@author: nknezek
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import scipy.special as sp

class Parameters(object):
    def __init__(self, source):
        self.source = source

class Solubility_calc():
    def __init__(self, element=None):
        self.element = element
        m_Mg = 24.305
        m_Si = 28.0855
        m_Fe = 55.845
        m_O = 15.9994
        self.m_Mg = m_Mg
        self.m_Si = m_Si
        self.m_Fe = m_Fe
        self.m_O = m_O
        self.m_i_c = np.array([m_Mg, m_Si, m_Fe, m_O])
        self.m_MgO = m_Mg + m_O
        self.m_SiO2 = m_Si + 2*m_O
        self.m_FeO = m_Fe + m_O
        self.m_MgSiO3 = m_Mg + m_Si + 3*m_O
        self.m_FeSiO3 = m_Fe + m_Si + 3*m_O
        self.m_i_m = np.array([self.m_MgO, self.m_SiO2, self.m_FeO, self.m_MgSiO3, self.m_FeSiO3])
        self.mass = None

    def solubility(self,P,T):
        raise NotImplementedError("must have a function to calculate solubility")

    def exsolution(self, solubilities, times, initial_concentration):
        sol_tmp = np.array(solubilities)
        sol_tmp[solubilities>initial_concentration] = initial_concentration
        return np.diff(sol_tmp)/np.diff(times)

class MgDubrovinskaia(Solubility_calc):

    def __init__(self):
        self.element = "Mg"
        data = np.recfromcsv('MgSolubility_Dubrovinskaia.csv')
        self.P_dat = data['p_gpa']
        self.E_dat = data['e_activation_energy']
        self.A_dat = data['a_const']

    def solubility(self, P, T, deg=2, P_dat = None, A_dat=None, E_dat=None):
        '''
        Calculates maximum solubility of Mg based on Sarah's parameterization of Dubrovinskaia's data points and a default deg=2 (polynomial) fit.

        First, calculates solubility at all pressure is given by data using X = A*exp(-E/(RT)), then fits the curve in pressure-space to find solubility at requested pressure.

        :param P: Pressure in [Pa]
        :param T: Temperature in [K]
        :param deg: degree of polynomial fit (default 2)
        :param P_dat: pressure array for data points
        :param A_dat: array of data points for A, the constant
        :param E_dat: array of data points for E, the activation energy
        :return: X: solubility of Mg
        '''
        if P_dat is None:
            P_dat = self.P_dat
        if A_dat is None:
            A_dat = self.A_dat
        if E_dat is None:
            E_dat = self.E_dat
        solubilities = np.zeros_like(T)
        for i, (P_val, T_val) in enumerate(zip(P,T)):
            X = self.solubility_at_P_dat(T_val, A_dat=A_dat, E_dat=E_dat)
            P_poly = np.polyfit(P_dat, X, deg)
            solubilities[i] = np.polyval(P_poly, P_val/1e6)
        return solubilities

    def exsolution_to_mol(self, exsolution):
        core_radius = 3480e3 # m
        core_volume = 4./3*np.pi*core_radius**3 # m^3
        core_density_avg = 1.09e4 # kg/m^3
        core_mass = core_volume*core_density_avg


    def solubility_at_P_dat(self, T, A_dat=None, E_dat=None):
        if A_dat is None:
            A_dat = self.A_dat
        if E_dat is None:
            E_dat = self.E_dat
        R=8.3144598
        X = A_dat*np.exp(-E_dat/(R*T))
        return X

    def plot_data_and_fit(self, T_list=None, P_fit=None, deg_fit=2, min_P_ind=0):
        if T_list is None:
            T_min = 4000.
            T_max = 8000.
            dT = 1000.
            T_list = np.linspace(T_min, T_max, (T_max-T_min)/dT+1)
        else:
            T_min = T_list[0]
            T_max = T_list[-1]
        if P_fit is None:
            P_fit = np.linspace(self.P_dat[min_P_ind]*1e6,135e6,100)
        jet = cm = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=T_min, vmax=T_max)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        lines = []
        fig = plt.figure(figsize=(8,6))
        for T in T_list:
            T_fit = T*np.ones_like(P_fit)
            X_fit = self.solubility(P_fit, T_fit, P_dat = self.P_dat[min_P_ind:], A_dat=self.A_dat[min_P_ind:], E_dat=self.E_dat[min_P_ind:], deg=deg_fit)
            colorVal = scalarMap.to_rgba(T)
            l, = plt.plot(P_fit/1e6, X_fit, '-', color=colorVal, label='{0:.0f}K'.format(T))
            lines.append(l)
            plt.plot(self.P_dat, self.solubility_at_P_dat(T), 'o', color=colorVal)
            plt.xlim(15,145)
            plt.ylim(0,1.0)
        plt.legend(handles=lines, loc=0)
        plt.title("Maximum Solubility of Mg in Fe metal at P,T")
        plt.xlabel('Pressure (GPa)')
        plt.ylabel('Solubility')
        plt.grid()
        plt.savefig('Mg_Sol_fit_deg{0}_min{1}Gpa.png'.format(deg_fit,self.P_dat[min_P_ind]))

class Badro(Solubility_calc):
    def __init__(self, element, a, b):
        super(Badro, self).__init__()
        self.element = element
        self.a = a
        self.b = b # [K]

    def equilibrium_constant(self, P, T):
        return np.exp(self.a - self.b/T)

class MgBadro(Badro):
    def __init__(self):
        super(MgBadro,self).__init__(element='Mg', a=1.23, b = 18816)
        self.mass = self.m_Mg

    def solubility(self,P,T, X_MgO=0.5, X_O=None, beta=None, a = None, b=None):
        '''
        Calculates Equilibrium solubility of Mg in core in mol frac, assuming a specified relationship of Oxygen to Mg

        :param P: Pressure (Pa)
        :param T: Temperature (K)
        :param beta: ratio of O to Mg in core (Badro assumes 1)
        :param X_MgO: mol frac of MgO in mantle (ratrio of Mg to Fe, ~0.9 in modern mantle)
        :param a: solubility fit parameter []
        :param b: solubility fit parameter [K]
        :param OxyType:
        :param X_O: mol frac of O in core (maybe 0.1 makes sense?)
        :return:
            X_Mg: saturated solubility of Mg in core [mol frac]
        '''

        if beta is not None and X_O is None:
            return self.solubility_OxyRatio(T, beta=beta, X_MgO=X_MgO, a=a, b=b)
        elif X_O is not None and beta is None:
            return self.solubility_OxyConst(T, X_O, X_MgO=X_MgO, a=a, b=b)
        else:
            raise ValueError('must specify either an O/Mg ratio (beta~1) or O mol frac (X_O~0.1) but not both')

    def solubility_OxyRatio(self, T, beta=1, X_MgO=0.5, a = None, b=None):
        '''
        Calculates Equilibrium solubility of Mg in core in mol frac, assuming a specified ratio of Oxygen to Mg

        :param T: temperature (K)
        :param beta: ratio of oxygen to Mg mol frac in core (Badro assumes 1)
        :param X_MgO: mol frac of MgO in silicate (Mg/Fe ratio) probably ~0.9 in modern mantle
        :param a: solubility fit parameter []
        :param b: solubility fit parameter [K]
        :return:
            X_Mg: saturated solubility of Mg in core [mol frac]
        '''
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return beta**(-0.5) * X_MgO**0.5 * np.power(10., (a - b/T)*0.5)

    def solubility_OxyConst(self, T, X_O, X_MgO=0.5, a=None, b=None):
        '''
        Calculates Equilibrium solubility of Mg in core in mol frac, assuming a specified amount of Oxygen

        :param T: temperature (K)
        :param X_O: mol frac of O in core
        :param X_MgO: mol frac of MgO in silicate (ratio Mg/Fe) probably ~0.9 in modern mantle
        :param a: solubility fit parameter []
        :param b: solubility fit parameter [K]
        :return:
            X_Mg: saturated solubility of Mg in core [mol frac]

        '''
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return X_MgO/X_O * np.power(10., a - b/T)

    def exsolution(self, P, T, X_MgO=0.5, X_O=None, beta=None, a=None, b=None):
        '''
        Calculates exsolution per K of Mg in core in mol frac, assuming a specified relationship of Oxygen to Mg

        :param P: Pressure (Pa)
        :param T: Temperature (K)
        :param beta: ratio of O to Mg in core (Badro assumes 1)
        :param X_MgO: mol frac of MgO in mantle (ratrio of Mg to Fe, ~0.9 in modern mantle)
        :param a: solubility fit parameter []
        :param b: solubility fit parameter [K]
        :param OxyType:
        :param X_O: mol frac of O in core (maybe 0.1 makes sense?)
        :return:
            X_Mg: saturated solubility of Mg in core [mol frac]
        '''
        if b is None:
            b = self.b
        return b/T**2*np.log(10)*self.solubility(P, T, X_MgO=X_MgO, X_O=X_O, beta=beta, a=a, b=b)

    def C_m(self, P, T, X_MgO=0.5, X_O=None, beta=None, a=None, b=None):
        '''
        Calculates exsolution per K of Mg in core in wt%, assuming a specified relationship of Oxygen to Mg

        :param P: Pressure (Pa)
        :param T: Temperature (K)
        :param beta: ratio of O to Mg in core (Badro assumes 1)
        :param X_MgO: mol frac of MgO in mantle (ratrio of Mg to Fe, ~0.9 in modern mantle)
        :param a: solubility fit parameter []
        :param b: solubility fit parameter [K]
        :param OxyType:
        :param X_O: mol frac of O in core (maybe 0.1 makes sense?)
        :return:
            C_m: exsolution per K of MgO in wt % at specified P, T
        '''
        return self.Mg_mol_frac_to_MgO_wtp(self.exsolution(P, T, X_MgO=X_MgO, X_O=X_O, beta=beta, a=a, b=b))

class AlBadro(Badro):
    def __init__(self):
        super(AlBadro,self).__init__(element='Al', a=4.1, b=36469)

    def solubility(self,P,T):
        return self.solubility_OxyRatio(T)

    def solubility_OxyRatio(self, T, beta=1, X_AlO15=0.015, a = None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return (beta**(-0.6))*(X_AlO15**0.4)*np.power(10., (a - b/T)*0.4)

    def solubility_OxyConst(self, T, X_O, X_AlO15=0.015, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return X_AlO15/X_O**1.5 *  np.power(10., a - b/T)

class Reactions():
    def __init__(self, params = None):
        if params is None:
            params = Parameters('Mantle reaction layer parameters')
        self.params = params
        self.params.reactions = Parameters('Mantle reaction layer parameters')
        pr = self.params.reactions
        pr.species_names = ['Mg','Si','Fe','O','core','MgO','SiO2','FeO','MgSiO3','FeSiO3','mantle']
        m_Mg = 24.305
        m_Si = 28.0855
        m_Fe = 55.845
        m_O = 15.9994
        pr.m_Mg = m_Mg
        pr.m_Si = m_Si
        pr.m_Fe = m_Fe
        pr.m_O = m_O
        pr.m_MgO = m_Mg + m_O
        pr.m_SiO2 = m_Si + 2*m_O
        pr.m_FeO = m_Fe + m_O
        pr.m_MgSiO3 = m_Mg + m_Si + 3*m_O
        pr.m_FeSiO3 = m_Fe + m_Si + 3*m_O
        pr.m_i_m = np.array([pr.m_MgO, pr.m_SiO2, pr.m_FeO, pr.m_MgSiO3, pr.m_FeSiO3])
        pr.m_i_c = np.array([pr.m_Mg, pr.m_Si, pr.m_Fe, pr.m_O])
        Cyr2s = 365.25*24*3600

        pr.thickness = 300 # [m] thickness of layer
        pr.density = 5500 # [kg/m^3] average density of layer in lower mantle

        pr.time_overturn = 800e6*Cyr2s # [s] overturn time of layer
        pr.V_c = 4/3*np.pi*3480e3**3 # [m^3] volume of total core
        pr.V_l = 4/3*np.pi*(3480e3+pr.thickness)**3 - pr.V_c # [m^3] volume of layer
        pr.d = 1. # [-] exponent of layer overturn expression
        pr.tau = pr.time_overturn # [s] constant of layer overturn expression
        pr.P = 135e9 # [Pa] pressure at CMB
        pr.time_solidify = 610e6*Cyr2s
        pr.dKMgSiO3_KMgSiO3 = 0.0 # these should be zero as no d/dT
        pr.dKFeSiO3_KFeSiO3 = 0.0 # these should be zero as no d/dT

        # initial (background mantle) compositions
        X_Mg_0 = 0.01
        X_Si_0 = 0.01
        X_O_0 = 0.15
        X_Fe_0 = 1 - X_Mg_0 - X_Si_0 - X_O_0

        X_MgO_0 = 0.02
        X_SiO2_0 = 0.02
        X_FeO_0 = 0.01
        X_MgSiO3_0 = (1-X_MgO_0-X_SiO2_0-X_FeO_0)/2
        X_FeSiO3_0 = (1-X_MgO_0-X_SiO2_0-X_FeO_0)/2

        X_i_m = [X_MgO_0, X_SiO2_0, X_FeO_0, X_MgSiO3_0, X_FeSiO3_0]
        X_i_c = [X_Mg_0, X_Si_0, X_O_0, X_Fe_0]
        M_i_m_0 = self.X2M_m(X_i_m)
        M_i_c_0 = self.X2M_c(X_i_c)
        pr.Moles_0 = M_i_c_0+M_i_m_0

    def wtp_i_c_2_wt_i_c(self, wtp_i_c, mass_c):
        return wtp_i_c*mass_c

    def wtp_i_c_2_M_i_c(self, wtp_i_c, mass_c):
        pr = self.params.reactions
        return wtp_i_c*mass_c/pr.m_i_c

    def Moles2wtp(self, Moles, calculate_Cs=False):
        pr = self.params.reactions
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = self.unwrap_Moles(Moles)
        wt_i_c = np.array([M_Mg, M_Si, M_Fe, M_O])*pr.m_i_c
        wt_c = np.sum(wt_i_c)
        wtp_i_c = wt_i_c/wt_c

        wt_i_m = np.array([M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3])*pr.m_i_m
        wt_m = np.sum(wt_i_m)
        wtp_i_m = wt_i_m/wt_m
        if calculate_Cs:
            wtp_MgO_c = pr.m_MgO*M_Mg/wt_c
            wtp_SiO2_c = pr.m_SiO2*M_Si/wt_c
            wtp_FeO_c = pr.m_FeO*M_Fe/wt_c
        return wtp_i_c, wtp_i_m, [wtp_MgO_c, wtp_SiO2_c, wtp_FeO_c]

    def unwrap_Moles(self, Moles):
        ''' helper function to unwrap mol values from Moles'''
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        return M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m

    def compute_dKs_dT(self, T_cmb, Moles):
        dKMgO_dT_KMgO = self.compute_dKMgO_dT_KMgO(T_cmb, Moles)
        dKSiO2_dT_KSiO2 = self.compute_dKSiO2_dT_KSiO2(T_cmb, Moles)
        dKFeO_dT_KFeO = self.compute_dKFeO_dT_KFeO(T_cmb, Moles)
        dKMgSiO3_dT_KMgSiO3 = self.compute_dKMgSiO3_dT_KMgSiO3(T_cmb, Moles)
        dKFeSiO3_dT_KFeSiO3 = self.compute_dKFeSiO3_dT_KFeSiO3(T_cmb, Moles)
        dKs_dT = [dKMgO_dT_KMgO, dKSiO2_dT_KSiO2, dKFeO_dT_KFeO, dKMgSiO3_dT_KMgSiO3, dKFeSiO3_dT_KFeSiO3]
        return dKs_dT

    def compute_dKMgO_dT_KMgO(self, T_cmb, Moles):
        ''' helper function to call Tushar's KD_MgO function

        :param T_cmb:
        :param Moles:
        :return:
        '''
        KMgO, dKMgO_dT = self.func_KD_MgO_val(T_cmb)
        return dKMgO_dT / KMgO

    def compute_dKSiO2_dT_KSiO2(self, T_cmb, Moles):
        '''helper function to call Tushar's func_KD_SiO2

        :param T_cmb:
        :param Moles:
        :return:
        '''
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = self.unwrap_Moles(Moles)
        X_Si = M_Si/M_c
        X_O = M_O/M_c
        KSiO2, dKSiO2_dT = self.func_KD_SiO2_val(X_Si, X_O, T_cmb)
        return dKSiO2_dT / KSiO2

    def compute_dKFeO_dT_KFeO(self, T_cmb, Moles):
        ''' helper function to call Tushar's func_KD_FeO'''
        KFeO, dKFeO_dT = self.func_KD_FeO_val(T_cmb)
        return dKFeO_dT / KFeO

    def compute_dKMgSiO3_dT_KMgSiO3(self, T_cmb, Moles):
        '''computes dK_MgSiO3, normally = 0

        :param T_cmb:
        :param Moles:
        :return:
        '''
        pr = self.params.reactions
        dKMgSiO3_KMgSiO3 = pr.dKMgSiO3_KMgSiO3
        return dKMgSiO3_KMgSiO3

    def compute_dKFeSiO3_dT_KFeSiO3(self, T_cmb, Moles):
        '''computes dK_FeSiO3, normally = 0

        :param T_cmb:
        :param Moles:
        :return:
        '''
        pr = self.params.reactions
        dKFeSiO3_KFeSiO3 = pr.dKFeSiO3_KFeSiO3
        return dKFeSiO3_KFeSiO3

    def func_KD_SiO2_val(self, X_Si, X_O, T_inp_base, P_inp_base=139e6, temp_diff_pm=10):
        '''
        K_D for SiO2 value
        Needs the following inputs : T_inp (Temperature in K), P_inp (Pressure in GPa),
        X_Si - mole frac Si in the core
        X_O - mole frac O in the core
        diff_pm = value of temp diff to calculate local deriv
        Note - (all the Mg related terms are zero since no data)
        '''
        P_inp = P_inp_base / 1e6  # convert to GPa
        ### Fit values from Hirose et al. 2017 paper (Eqn 5 in the Supplementary material)
        fit_KD_FeO_a = 0.3009  # (+/- 0.1120)
        fit_KD_FeO_b = 0
        fit_KD_FeO_c = -36.8332  # (+/- 5.5957)
        # Fit values from Rebecca Fischer et al. 2015 (extended Data Table 1 - Hirose 2017)
        fit_KD_Si_a = 1.3  # (+/- 0.3)
        fit_KD_Si_b = -13500  # (+/- 900)
        fit_KD_Si_c = 0
        ### Acitivity coeff fit values - Hirose et al. 2017, extended Data Table 1
        epsf_OO = -9.16  # (+/- 4.27)
        epsf_OSi = 7.73  # (+/- 4.53)
        epsf_SiSi = 0

        def func_KD(T_inp):
            log_KD_Feo = fit_KD_FeO_a + fit_KD_FeO_b / T_inp + fit_KD_FeO_c * P_inp / T_inp
            log_KD_Si = fit_KD_Si_a + fit_KD_Si_b / T_inp + fit_KD_Si_c * P_inp / T_inp

            # Steelmaking Handbook method for correcting gamma used:
            ##  (activity)  log(gamma(T)) = Tr/T*log(gamma(Tr))
            T_ref = 1873  # Kelvin, Hirose 2017

            #### Activity coeff values for gamma_Si (based on Hirose et al. 2017 + Ma 2001 Eqn 23-24 in the paper )
            ## Since the cross term due to Si-Si does not contribute,
            # so the only term that contributes for gamma_Si is the Si-O term (all the Mg related terms are zero since no data)
            sum_v = epsf_OSi * (X_O * (1. + np.log(1. - X_O) / X_O - 1. / (1. - X_O)) - X_O ** 2. * X_Si * (
            1. / (1. - X_Si) + 1. / (1. - X_O) + X_Si / (2. * (1. - X_Si) ** 2.) - 1.)) / 2.303
            log_gam_Si = -(T_ref / T_inp) * (epsf_SiSi * np.log(1. - X_Si) / 2.303 + sum_v)
            del sum_v
            #### Activity coeff values for gamma_O (based on Hirose et al. 2017 + Ma 2001 Eqn 23-24 in the paper )
            # Note - all the Mg related terms are zero since no data
            sum_v = epsf_OSi * (X_Si * (1. + np.log(1. - X_Si) / X_Si - 1. / (1. - X_Si)) - X_Si ** 2. * X_O * (
            1. / (1. - X_O) + 1. / (1. - X_Si) + X_O / (2. * (1. - X_O) ** 2.) - 1.)) / 2.303
            log_gam_O = -(T_ref / T_inp) * (epsf_OO * np.log(1. - X_O) / 2.303 + sum_v)
            del sum_v
            KD_SiO2 = (10. ** log_KD_Si) * ((10. ** log_KD_Feo) ** 2.) / (10. ** (log_gam_Si)) / (10. ** (
            log_gam_O)) ** 2.
            return KD_SiO2

        KD_SiO2 = func_KD(T_inp_base)
        KD_SiO2_T_deriv = (func_KD(T_inp_base + temp_diff_pm) - func_KD(T_inp_base - temp_diff_pm)) / (
        2. * temp_diff_pm)
        return KD_SiO2, KD_SiO2_T_deriv

    def func_KD_FeO_val(self, T_inp, P_inp_base=139e6):
        '''
        K_D for FeO value
        Needs the following inputs : T_inp (Temperature in K), P_inp (Pressure in GPa),
        '''
        P_inp = P_inp_base / 1e6  # convert to GPa
        ### Fit values from Hirose et al. 2017 paper (Eqn 5 in the Supplementary material)
        fit_KD_FeO_a = 0.3009  # (+/- 0.1120)
        fit_KD_FeO_b = 0
        fit_KD_FeO_c = -36.8332  # (+/- 5.5957)
        log_KD_Feo = fit_KD_FeO_a + fit_KD_FeO_b / T_inp + fit_KD_FeO_c * P_inp / T_inp
        KD_FeO = 10. ** log_KD_Feo
        KD_FeO_Tderiv = (KD_FeO) * -1. * fit_KD_FeO_c * P_inp * np.log(10.) / T_inp ** 2.
        return KD_FeO, KD_FeO_Tderiv

    def func_KD_MgO_val(self, T_inp, P_inp_base=139e6):
        '''
        K_D for MgO value
        Needs the following inputs : T_inp (Temperature in K), P_inp (Pressure in GPa),
        '''
        P_inp = P_inp_base / 1e6  # convert to GPa
        ### Fit values from Badro et al. 2015 paper (Eqn 5 in the Supplementary material)
        fit_KD_MgO_a = 1.23  # (+/- 0.7)
        fit_KD_MgO_b = -18816  # (+/- 2600)
        fit_KD_MgO_c = 0
        log_KD_Mgo = fit_KD_MgO_a + fit_KD_MgO_b / T_inp + fit_KD_MgO_c * P_inp / T_inp
        KD_MgO = 10. ** log_KD_Mgo
        KD_MgO_Tderiv = (KD_MgO) * -1. * fit_KD_MgO_b * np.log(10.) / T_inp ** 2.
        return KD_MgO, KD_MgO_Tderiv

    def unwrap_dKs_dT(self, dKs_dT):
        ''' helper function to unwrap dK values from dKs_dT'''
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs_dT
        return dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3

    def compute_Moles_background(self, fraction_MgFe, X_MgFeO, X_SiO2, M_m=None):
        X_MgO = X_MgFeO*fraction_MgFe
        X_FeO = X_MgFeO*(1-fraction_MgFe)
        X_MgFeSiO3 = (1-X_SiO2-X_MgFeO)
        X_MgSiO3 = X_MgFeSiO3*fraction_MgFe
        X_FeSiO3 = X_MgFeSiO3*(1-fraction_MgFe)
        if M_m is None:
            pr = self.params.reactions
            V_l = pr.V_l
            density = pr.density
            mass_l = density * V_l  # [kg]
            X_i_m = np.array([X_MgO, X_SiO2, X_FeO, X_MgSiO3, X_FeSiO3])
            M_m = mass_l / np.sum(pr.m_i_m * X_i_m)
        Moles = list(np.array([0,0,0,0,0, X_MgO, X_SiO2, X_FeO, X_MgSiO3, X_FeSiO3, 1])*M_m)
        return Moles

    def erode_term(self, M_i, M_i_0, d=None, tau=None):
        ''' Layer erosion term given current and initial number of moles of species i and initial total moles in the layer

        :param M_i:
        :param M_i_0:
        :param M_m_0:
        :return:
        '''
        pr =self.params.reactions
        if d is None:
            d = pr.d
        if tau is None:
            tau = pr.tau
        return np.sign(M_i_0 - M_i) * M_i_0 / tau * ((np.abs(M_i - M_i_0) / M_i_0 + 1)**d-1)

    def X2M_m(self, X_i_m):
        '''convert Mole Fractions to Moles for the mantle reaction layer

        :param X_i_m_0:
        :param X_i_c_0:
        :return:
        '''
        pr =self.params.reactions
        V_l = pr.V_l
        thickness = pr.thickness
        density = pr.density

        # make sure mole fractions actually add to 1
        X_i_m = X_i_m/np.sum(X_i_m)
        # X_i_c = X_i_c_0 / np.sum(X_i_c_0)

        mass_l = density*V_l # [kg]
        M_m = mass_l / np.sum(pr.m_i_m*X_i_m)
        M_i_m = list(M_m*X_i_m)
        M_i_m.append(M_m)
        # mass_c = 1.94e24 # [kg] mass of core
        # M_c = mass_c / np.sum(pr.m_i_c*X_i_c)
        # M_i_c = list(M_c*X_i_c)
        # M_i_c.append(M_c)
        return M_i_m

    def X2M_c(self, X_i_c):
        '''convert Mole Fractions to Moles for the mantle reaction layer

        :param X_i_m_0:
        :param X_i_c_0:
        :return:
        '''
        pr =self.params.reactions
        V_c = pr.V_c
        thickness = pr.thickness
        density = pr.density

        # make sure mole fractions actually add to 1
        X_i_c = X_i_c / np.sum(X_i_c)

        mass_c = 1.94e24 # [kg] mass of core
        M_c = mass_c / np.sum(pr.m_i_c*X_i_c)
        M_i_c = list(M_c*X_i_c)
        M_i_c.append(M_c)
        return M_i_c

    def dMi_b(self, Moles=None, dTdt=None):
        '''compute the erosion term incorporated directly into the equations

        :param Moles:
        :param dTdt:
        :return:
        '''
        pr = self.params.reactions
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = self.unwrap_Moles(Moles)
        M_Mg_b, M_Si_b, M_Fe_b, M_O_b, M_c_b, M_MgO_b, M_SiO2_b, M_FeO_b, M_MgSiO3_b, M_FeSiO3_b, M_m_b = self.unwrap_Moles(pr.Moles_0)
        dM_MgO_dt_b = -self.erode_term(M_MgO, M_MgO_b)/dTdt
        dM_SiO2_dt_b = -self.erode_term(M_SiO2, M_SiO2_b)/dTdt
        dM_FeO_dt_b = -self.erode_term(M_FeO, M_FeO_b)/dTdt
        dM_MgSiO3_dt_b = -self.erode_term(M_MgSiO3, M_MgSiO3_b)/dTdt
        dM_FeSiO3_dt_b = -self.erode_term(M_FeSiO3, M_FeSiO3_b)/dTdt

        # mantle visibility correction
        tau_m = pr.tau/100
        dM_MgO_dt_b += -self.erode_term(M_m, M_m_b, tau=tau_m)/dTdt*M_MgO/M_m
        dM_SiO2_dt_b += -self.erode_term(M_m, M_m_b, tau=tau_m)/dTdt*M_SiO2/M_m
        dM_FeO_dt_b += -self.erode_term(M_m, M_m_b, tau=tau_m)/dTdt*M_FeO/M_m
        dM_MgSiO3_dt_b += -self.erode_term(M_m, M_m_b, tau=tau_m)/dTdt*M_MgSiO3/M_m
        dM_FeSiO3_dt_b += -self.erode_term(M_m, M_m_b, tau=tau_m)/dTdt*M_FeSiO3/M_m
        return [0,0,0,0,0,dM_MgO_dt_b, dM_SiO2_dt_b, dM_FeO_dt_b, dM_MgSiO3_dt_b, dM_FeSiO3_dt_b,0]

    def dMi_dt_erode(self, Moles):
        '''calculate the erosion rate for each molar species in the mantle layer

        :param Moles:
        :return:
        '''
        pr = self.params.reactions

        # calculate the background mantle composition with proper Mg-Fe fraction
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = self.unwrap_Moles(Moles)
        fraction_MgFe = (M_MgO/(M_MgO+M_FeO) + M_MgSiO3/(M_MgSiO3+M_FeSiO3))/2
        M_Mg_0, M_Si_0, M_Fe_0, M_O_0, M_c_0, M_MgO_0, M_SiO2_0, M_FeO_0, M_MgSiO3_0, M_FeSiO3_0, M_m_0 = self.unwrap_Moles(pr.Moles_0)
        fraction_MgFe_0 = (M_MgO_0/(M_MgO_0+M_FeO_0) + M_MgSiO3_0/(M_MgSiO3_0+M_FeSiO3_0))/2
        fraction_MgFe_b = (fraction_MgFe_0+fraction_MgFe)/2
        X_MgFeO_0 = (M_MgO_0+M_FeO_0)/M_m_0
        Moles_b = self.compute_Moles_background(fraction_MgFe_b, X_MgFeO_0, M_SiO2_0/M_m_0, M_m_0)
        M_Mg_b, M_Si_b, M_Fe_b, M_O_b, M_c_b, M_MgO_b, M_SiO2_b, M_FeO_b, M_MgSiO3_b, M_FeSiO3_b, M_m_b = self.unwrap_Moles(Moles_b)

        # compute the erosional terms
        dM_MgO_dt_e = self.erode_term(M_MgO, M_MgO_b, M_m_b)
        dM_SiO2_dt_e = self.erode_term(M_SiO2, M_SiO2_b, M_m_b)
        dM_FeO_dt_e = self.erode_term(M_FeO, M_FeO_b, M_m_b)
        dM_MgSiO3_dt_e = self.erode_term(M_MgSiO3, M_MgSiO3_b, M_m_b)
        dM_FeSiO3_dt_e = self.erode_term(M_FeSiO3, M_FeSiO3_b, M_m_b)
        return [dM_MgO_dt_e, dM_SiO2_dt_e, dM_FeO_dt_e, dM_MgSiO3_dt_e, dM_FeSiO3_dt_e]

    def dMoles_dT(self, Moles=None, T_cmb=None, dKs_dT=None, dTdt=None, dMi_b=None):
        '''calcluate the change in Moles vs temperature T for each molar species in the core and mantle

        :param Moles:
        :param T_cmb:
        :param dKs_dT:
        :return:
        '''
        if dKs_dT is None:
            dKs_dT = self.compute_dKs_dT(Moles=Moles, T_cmb=T_cmb)
        if dMi_b is None:
            dMi_b = self.dMi_b(Moles=Moles, dTdt=dTdt)

        # core
        dM_Mg_dT = self.dM_Mg_dTc(Moles, dKs_dT, dMi_b)
        dM_Si_dT = self.dM_Si_dTc(Moles, dKs_dT, dMi_b)
        dM_Fe_dT = self.dM_Fe_dTc(Moles, dKs_dT, dMi_b)
        dM_O_dT = self.dM_O_dTc(Moles, dKs_dT, dMi_b)
        dM_c_dT = np.sum([dM_Mg_dT, dM_Si_dT, dM_Fe_dT, dM_O_dT])

        # mantle
        dM_MgO_dT = self.dM_MgO_dTc(Moles, dKs_dT, dMi_b)
        dM_SiO2_dT = self.dM_SiO2_dTc(Moles, dKs_dT, dMi_b)
        dM_FeO_dT = self.dM_FeO_dTc(Moles, dKs_dT, dMi_b)
        dM_MgSiO3_dT = self.dM_MgSiO3_dTc(Moles, dKs_dT, dMi_b)
        dM_FeSiO3_dT = self.dM_FeSiO3_dTc(Moles, dKs_dT, dMi_b)
        dM_m_dT = np.sum([dM_MgO_dT, dM_FeO_dT, dM_SiO2_dT, dM_MgSiO3_dT, dM_FeSiO3_dT])
        return [dM_Mg_dT, dM_Si_dT, dM_Fe_dT, dM_O_dT, dM_c_dT, dM_MgO_dT, dM_SiO2_dT, dM_FeO_dT, dM_MgSiO3_dT,
                dM_FeSiO3_dT, dM_m_dT]

    def dMoles_dt(self, Moles=None, T_cmb=None, dTc_dt=None, dKs_dT=None, dMoles_dT=None):
        '''calculate the change in Moles vs time (t) for each molar species in the core and mantle

        :param Moles:
        :param T_cmb:
        :param dTc_dt:
        :return:
        '''
        if dMoles_dT is None:
            dMoles_dT = self.dMoles_dT(Moles=Moles, T_cmb=T_cmb, dKs_dT=dKs_dT, dTdt=dTc_dt)

        dM_Mg_dT, dM_Si_dT, dM_Fe_dT, dM_O_dT, dM_c_dT, dM_MgO_dT, dM_SiO2_dT, \
                dM_FeO_dT, dM_MgSiO3_dT, dM_FeSiO3_dT, dM_m_dT = self.unwrap_Moles(dMoles_dT)
        # core
        dM_Mg_dt = dM_Mg_dT*dTc_dt
        dM_Si_dt = dM_Si_dT*dTc_dt
        dM_Fe_dt = dM_Fe_dT*dTc_dt
        dM_O_dt = dM_O_dT*dTc_dt
        dM_c_dt = np.sum([dM_Mg_dt, dM_Si_dt, dM_Fe_dt, dM_O_dt])

        # mantle
        dM_MgO_dt = dM_MgO_dT*dTc_dt
        dM_SiO2_dt = dM_SiO2_dT * dTc_dt
        dM_FeO_dt = dM_FeO_dT * dTc_dt
        dM_MgSiO3_dt = dM_MgSiO3_dT * dTc_dt
        dM_FeSiO3_dt = dM_FeSiO3_dT * dTc_dt
        dM_m_dt = np.sum([dM_MgO_dt, dM_SiO2_dt, dM_FeO_dt, dM_MgSiO3_dt, dM_FeSiO3_dt])
        return [dM_Mg_dt, dM_Si_dt, dM_Fe_dt, dM_O_dt, dM_c_dt, dM_MgO_dt, dM_SiO2_dt, dM_FeO_dt, dM_MgSiO3_dt, dM_FeSiO3_dt, dM_m_dt]

    def dM_MgSiO3_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_MgSiO3 * (M_Fe * (M_O * (M_FeO * (M_MgO * (M_SiO2 * (
        4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                            -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                   -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
        M_SiO2 * (4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                                         M_SiO2 * (
                                                                                                         4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                                         4.0 * dM_FeO_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                         -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er))) + dKFeO_KFeO * (
                                   M_Mg * M_O * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                   M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                   -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Si * (M_Mg * (
                                   M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                   M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                   M_FeO * (2.0 * M_SiO2 + 10.0 * M_m) + M_FeSiO3 * (
                                   -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m))) + M_MgO * M_O * (M_FeO * (
                                   3.0 * M_SiO2 + 6.0 * M_m) + M_FeSiO3 * (-6.0 * M_SiO2 + 15.0 * M_m))) + M_c * (
                                   M_Mg * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                   M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) + M_O * (
                                   M_FeO + M_MgO - M_m) - M_SiO2 * M_m)) + M_MgO * M_O * (
                                   -M_FeO * M_SiO2 - M_FeSiO3 * M_m) + M_Si * (M_Mg * (
                                   M_FeO * (-M_SiO2 - 3.0 * M_m) + M_FeSiO3 * (
                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                   M_FeO + M_FeSiO3)) + M_MgO * (M_FeO * (-2.0 * M_SiO2 - 2.0 * M_m) + M_FeSiO3 * (
                                   2.0 * M_SiO2 - 6.0 * M_m)))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (M_O * (
        M_Fe * M_FeO * M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_Mg * (M_Fe * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
        -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeO * M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m))) + M_Si * (
                                                                                                   M_Mg * (M_Fe * (
                                                                                                   M_FeO * (
                                                                                                   -M_SiO2 + M_m) + M_MgO * (
                                                                                                   -M_SiO2 + M_m) + M_O * (
                                                                                                   -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_SiO2 * M_m) + M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -M_SiO2 + M_m) + M_O * (
                                                                                                           -9.0 * M_MgO - 6.0 * M_SiO2 + 15.0 * M_m))) + M_MgO * (
                                                                                                   M_Fe * (M_FeO * (
                                                                                                   -M_SiO2 + M_m) + M_O * (
                                                                                                           -9.0 * M_FeO - 6.0 * M_SiO2 + 15.0 * M_m)) + M_FeO * M_O * (
                                                                                                   -9.0 * M_SiO2 + 9.0 * M_m))) + M_c * (
                                                                                                   M_Mg * (M_Fe * (
                                                                                                   M_FeO * (
                                                                                                   M_SiO2 - M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_O * (
                                                                                                   M_FeO + M_MgO - M_m) - M_SiO2 * M_m) + M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           M_SiO2 - M_m) + M_O * (
                                                                                                           M_MgO - M_m))) + M_MgO * (
                                                                                                   M_Fe * (M_FeO * (
                                                                                                   M_SiO2 - M_m) + M_O * (
                                                                                                           M_FeO - M_m)) + M_FeO * M_O * (
                                                                                                   M_SiO2 - M_m)) + M_Si * (
                                                                                                   M_Mg * (M_Fe * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_O + M_SiO2 - 9.0 * M_m) + M_FeO * (
                                                                                                           4.0 * M_MgO + 2.0 * M_SiO2 - 6.0 * M_m)) + M_MgO * (
                                                                                                   M_Fe * (
                                                                                                   4.0 * M_FeO + 2.0 * M_SiO2 - 6.0 * M_m) + M_FeO * (
                                                                                                   4.0 * M_SiO2 - 4.0 * M_m))))) + M_Mg * (
                           M_O * (M_Fe * (M_FeO * (M_SiO2 * (
                           4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                   -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_MgO * (
                                          M_SiO2 * (
                                          4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                          -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                          -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_FeO * (
                                  M_MgO * (M_SiO2 * (
                                  4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                           -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                  -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                           M_SiO2 * (4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                           -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                           4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                              4.0 * dM_FeO_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                        -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er))) + dKMgO_KMgO * (
                           M_Fe * M_O * (-4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                           M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Si * (M_Fe * (
                           -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                           M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                           M_FeO * (M_SiO2 - 4.0 * M_m) + M_FeSiO3 * (
                           9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m) + M_MgO * (
                           3.0 * M_SiO2 + 6.0 * M_m) - 9.0 * M_SiO2 * M_m)) + M_O * (M_FeO * (
                           M_MgO * (3.0 * M_SiO2 + 6.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                           9.0 * M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                  3.0 * M_SiO2 + 6.0 * M_m) - 9.0 * M_SiO2 * M_m))) + M_c * (
                           M_Fe * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                           M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                   M_FeSiO3 * (-M_FeO - M_MgO + M_m) + M_SiO2 * (-M_FeO - M_MgO + M_m))) + M_O * (
                           M_FeO * M_SiO2 * (-M_MgO + M_m) + M_FeSiO3 * (
                           M_FeO * (-M_SiO2 + M_m) + M_SiO2 * (-M_MgO + M_m))) + M_Si * (M_Fe * (
                           M_FeO * (-M_SiO2 + M_m) + M_FeSiO3 * (
                           -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_MgO * (
                           -2.0 * M_SiO2 - 2.0 * M_m) + M_O * (
                           -M_FeO - M_FeSiO3 - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_FeO * (M_MgO * (
                           -2.0 * M_SiO2 - 2.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                          -2.0 * M_SiO2 - 2.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                         M_FeO * (
                                                                                         -M_SiO2 + M_m) + M_FeSiO3 * (
                                                                                         -M_SiO2 + M_m)))))) + M_Si * (
                           M_Fe * (M_FeO * (M_MgO * (
                           M_SiO2 * (dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                           -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                            -dM_MgO_er - dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                           M_SiO2 * (dM_MgO_er + dM_MgSiO3_er) + M_m * (-dM_MgO_er - dM_MgSiO3_er)) + M_MgO * (
                                                                                      M_SiO2 * (
                                                                                      1.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                                                      1.0 * dM_FeO_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                      -dM_MgO_er - dM_MgSiO3_er)) + M_O * (
                                   M_FeO * (M_MgO * (
                                   6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                            dM_MgO_er + dM_MgSiO3_er) + M_m * (
                                            -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
                                   M_FeO * (9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_MgO * (
                                   6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 15.0 * dM_MgO_er + 24.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                   4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                                   -25.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                                   6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 18.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_m * (
                                                                                        -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                   -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er))) + M_Mg * (M_Fe * (M_FeO * (
                           M_SiO2 * (dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                           -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                           dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                    -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_O * (
                                                                                              M_FeO * (
                                                                                              6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_MgO * (
                                                                                              6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                              4.0 * dM_FeO_er + 10.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 10.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_m * (
                                                                                              -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                              -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_FeO * (
                                                                                      M_MgO * (M_SiO2 * (
                                                                                      dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                               -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                      -1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                                      M_FeO * (M_SiO2 * (
                                                                                      1.0 * dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                               -1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                      M_SiO2 * (
                                                                                      dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                      1.0 * dM_FeO_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                      -1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_O * (
                                                                                      M_FeO * (M_MgO * (
                                                                                      6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                               6.0 * dM_FeO_er + 12.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 10.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_m * (
                                                                                               -15.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                      M_FeO * (
                                                                                      -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er) + M_MgO * (
                                                                                      -3.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                      6.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 10.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_m * (
                                                                                      15.0 * dM_FeO_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er)))) + M_O * (
                           M_FeO * (M_MgO * (M_SiO2 * (
                           9.0 * dM_FeO_er + 18.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 18.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_m * (
                                             -9.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                    -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                           M_SiO2 * (9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_m * (
                           -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                           9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 18.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_m * (
                                                                              9.0 * dM_FeO_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                          -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er))) + dKSiO2_KSiO2 * (
                           M_O * (M_Fe * M_MgO * (
                           M_FeO * (-2.0 * M_SiO2 - 4.0 * M_m) + M_FeSiO3 * (4.0 * M_SiO2 - 10.0 * M_m)) + M_Mg * (
                                  M_Fe * (M_FeO * (-2.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                  -2.0 * M_SiO2 - 4.0 * M_m) + 6.0 * M_SiO2 * M_m) + M_FeO * (
                                  M_MgO * (-2.0 * M_SiO2 - 4.0 * M_m) + 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                  M_FeO * (-6.0 * M_SiO2 + 6.0 * M_m) + M_MgO * (
                                  -2.0 * M_SiO2 - 4.0 * M_m) + 6.0 * M_SiO2 * M_m))) + M_c * (M_Mg * (M_Fe * (
                           M_FeO * (M_SiO2 + M_m) + M_MgO * (M_SiO2 + M_m) + M_O * (
                           -M_FeO - M_MgO + M_m) - 2.0 * M_SiO2 * M_m) + M_FeO * (M_MgO * (
                           M_SiO2 + M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                           2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (M_SiO2 + M_m) - 2.0 * M_SiO2 * M_m) + M_O * (M_FeO * (
                           -M_MgO + M_m) + M_FeSiO3 * (-M_MgO + M_m))) + M_MgO * (M_Fe * (
                           M_FeO * (M_SiO2 + M_m) + M_FeSiO3 * (-M_SiO2 + 3.0 * M_m) + M_O * (
                           -M_FeO - M_FeSiO3 - M_SiO2 + M_m)) + M_O * (M_FeO * (-M_SiO2 + M_m) + M_FeSiO3 * (
                           -M_SiO2 + M_m)))))) + M_c * (M_Fe * (M_FeO * (M_MgO * (
        M_SiO2 * (-dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
        M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
        -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                            -dM_FeO_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                              dM_MgO_er + dM_MgSiO3_er)) + M_O * (
                                                                M_FeO * (M_MgO * (
                                                                -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_SiO2 * (
                                                                         -dM_MgO_er - dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                M_FeO * (-dM_MgO_er - dM_MgSiO3_er) + M_MgO * (
                                                                -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                dM_MgO_er + dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                                                                -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                      dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                dM_MgO_er + dM_MgSiO3_er))) + M_Mg * (M_Fe * (M_FeO * (
        M_SiO2 * (-dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgO * (M_SiO2 * (
        -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                              dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_O * (
                                                                                                              M_FeO * (
                                                                                                              -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                                                                                                              -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                              dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                              dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_FeO * (
                                                                                                      M_MgO * (
                                                                                                      M_SiO2 * (
                                                                                                      -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                                      dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                      dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      M_SiO2 * (
                                                                                                      -dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                                      dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                                      M_SiO2 * (
                                                                                                      -1.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                                      -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                      dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_O * (
                                                                                                      M_FeO * (M_MgO * (
                                                                                                      -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      dM_FeO_er + dM_FeSiO3_er) + M_MgO * (
                                                                                                      dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                      -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er)))) + M_O * (
                                                        M_FeO * (M_MgO * (M_SiO2 * (
                                                        -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                 dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                                                        M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                        dM_MgO_er + dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                                                        -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                              -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                          dM_MgO_er + dM_MgSiO3_er))) + M_Si * (
                                                        M_Fe * (M_FeO * (M_MgO * (
                                                        -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                         -dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                         dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                M_FeO * (
                                                                -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                                -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                -dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                M_SiO2 * (
                                                                -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_O * (
                                                                M_FeO * (-dM_MgO_er - dM_MgSiO3_er) + M_FeSiO3 * (
                                                                -dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                                                                -dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                dM_MgO_er + dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_FeO * (
                                                        M_MgO * (M_SiO2 * (
                                                        -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                 4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                        4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                                                        M_SiO2 * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                                        4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                                                        -4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                                                          -4.0 * dM_FeO_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                             4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_Mg * (
                                                        M_Fe * (M_FeO * (
                                                        -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_MgO * (
                                                                -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_FeO * (
                                                        M_MgO * (
                                                        -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                        -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                        6.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                        M_FeO * (4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_MgO * (
                                                        2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                        -2.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                        -6.0 * dM_FeO_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_O * (
                                                        M_FeO * (dM_FeO_er + dM_FeSiO3_er) + M_FeSiO3 * (
                                                        dM_FeO_er + dM_FeSiO3_er))) + M_O * (M_FeO * (
                                                        M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                        dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (M_SiO2 * (
                                                        -dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                                                 dM_MgO_er + dM_MgSiO3_er))))) + dKMgSiO3_KMgSiO3 * (
                           M_O * (M_Fe * M_MgO * (-4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                           M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Mg * (M_Fe * (M_FeSiO3 * (
                           M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                       -4.0 * M_FeO - 4.0 * M_MgO)) + M_MgO * (
                                                                                               -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                               M_FeO * (
                                                                                               4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)))) + M_Si * (
                           M_Mg * (M_Fe * (
                           M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                           M_FeO * (M_SiO2 - 4.0 * M_m) + M_FeSiO3 * (
                           9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m) + M_MgO * (
                           M_SiO2 - 4.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (
                                   -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                   M_FeO * (M_MgO * (M_SiO2 - 4.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                   M_FeO * (9.0 * M_MgO + 9.0 * M_SiO2 - 9.0 * M_m) + M_MgO * (
                                   M_SiO2 - 4.0 * M_m) - 9.0 * M_SiO2 * M_m))) + M_MgO * (M_Fe * (
                           -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                           M_FeO * (M_SiO2 - 4.0 * M_m) + M_FeSiO3 * (
                           9.0 * M_FeO + 4.0 * M_SiO2 - 25.0 * M_m) - 9.0 * M_SiO2 * M_m)) + M_O * (
                                                                                          -9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                          M_FeO * (
                                                                                          9.0 * M_SiO2 - 9.0 * M_m) - 9.0 * M_SiO2 * M_m)))) + M_c * (
                           M_Mg * (M_Fe * (
                           M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                           M_FeSiO3 * (-M_FeO - M_MgO + M_m) + M_SiO2 * (-M_FeO - M_MgO + M_m)) + M_SiO2 * M_m * (
                           M_FeO + M_MgO)) + M_MgO * (
                                   M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                   M_FeO * M_SiO2 * (-M_MgO + M_m) + M_FeSiO3 * (
                                   M_FeO * (-M_MgO - M_SiO2 + M_m) + M_SiO2 * (-M_MgO + M_m)))) + M_MgO * (M_Fe * (
                           M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                           M_FeSiO3 * (-M_FeO + M_m) + M_SiO2 * (-M_FeO + M_m))) + M_O * (
                                                                                                           M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m))) + M_Si * (
                           M_Mg * (M_Fe * (M_FeO * (-M_SiO2 + M_m) + M_FeSiO3 * (
                           -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_MgO * (-M_SiO2 + M_m) + M_O * (
                                           -M_FeO - M_FeSiO3 - M_MgO - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_FeO * (
                                   M_MgO * (-M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                   M_FeO * (-4.0 * M_MgO - 4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                   -M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_O * (
                                   M_FeO * (-M_MgO - M_SiO2 + M_m) + M_FeSiO3 * (-M_MgO - M_SiO2 + M_m))) + M_MgO * (
                           M_Fe * (M_FeO * (-M_SiO2 + M_m) + M_FeSiO3 * (-4.0 * M_FeO - M_SiO2 + 9.0 * M_m) + M_O * (
                           -M_FeO - M_FeSiO3 - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + 4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                           M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_O * (
                           M_FeO * (-M_SiO2 + M_m) + M_FeSiO3 * (-M_SiO2 + M_m))))))) / (M_O * (M_Fe * (M_MgO * (
        4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
        -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (
        M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
        -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
                                                                                         M_Fe * (M_MgO * (
                                                                                         M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (M_MgO * (
                                                                                                 -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -M_SiO2 + M_m) + M_MgO * (
                                                                                                 -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                                 M_MgO * (M_FeO * (
                                                                                                 -M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                                          -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                                 -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                                 -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
                                                                                         M_Fe * (M_FeSiO3 * (M_FeO * (
                                                                                         -M_SiO2 + M_m) + M_MgO * (
                                                                                                             -M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -M_SiO2 + M_m) + M_MgO * (
                                                                                                 -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                                                                 M_FeO * (
                                                                                                 -M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                                 -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                                 -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                                                                                                 -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                 M_FeO + M_MgO)) + M_MgO * (
                                                                                         M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (M_MgO * (
                                                                                         -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -M_SiO2 + M_m) + M_MgO * (
                                                                                         -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                         M_FeO * (M_MgO * (
                                                                                         -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                         -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                         M_FeO * (
                                                                                         -9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
                                                                                         -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (
                                                                                         M_MgO * (
                                                                                         9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (M_MgO * (
                                                                                         -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                         -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
                                                                                         M_Fe * (M_MgO * (
                                                                                         -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (M_MgO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 M_SiO2 - M_m) + M_MgO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                                 M_MgO * (M_FeSiO3 * (
                                                                                                 M_FeO - M_m) + M_SiO2 * (
                                                                                                          M_FeO - M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                                 M_FeO + M_MgO - M_m) + M_MgO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (
                                                                                         M_Fe * (M_FeSiO3 * (M_FeO * (
                                                                                         M_SiO2 - M_m) + M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 M_SiO2 - M_m) + M_MgO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                                                                 M_FeSiO3 * (
                                                                                                 M_FeO + M_MgO - M_m) + M_MgSiO3 * (
                                                                                                 M_FeO + M_MgO - M_m) + M_SiO2 * (
                                                                                                 M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (
                                                                                                 -M_FeO - M_MgO)) + M_MgO * (
                                                                                         -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (M_MgO * (
                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_SiO2 - M_m) + M_MgO * (
                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                         M_FeO * M_SiO2 * (
                                                                                         M_MgO - M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_MgO - M_m) + M_FeSiO3 * (
                                                                                         M_FeO + M_MgO - M_m)))) + M_O * (
                                                                                         M_MgO * (
                                                                                         -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (M_MgO * (
                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_SiO2 - M_m) + M_MgO * (
                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (
                                                                                         M_Fe * (M_MgO * (M_FeO * (
                                                                                         M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                          4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                 4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                 4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                 M_MgO * (
                                                                                                 M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                 M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                         M_Fe * (M_FeO * (
                                                                                         M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                 4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                 M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                 4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                 M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                         M_MgO * (
                                                                                         M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                         M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                         M_FeO * (
                                                                                         4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                         4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                         M_FeO * (
                                                                                         M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                         M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                         M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                         -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (M_MgO * (
                                                                                         4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                         4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                         M_MgO * (M_FeO * (
                                                                                         M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                  M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (
                                                                                         M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                         M_SiO2 - M_m))))))

    def dM_MgO_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_MgO * (M_Fe * (M_O * (M_FeO * M_SiO2 * M_m * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_FeSiO3 * (
        M_FeO * (M_SiO2 * (4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
        -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_SiO2 * M_m * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                                       M_FeO * (M_SiO2 * (
                                       -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                       M_SiO2 * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                       -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)))) + dKFeO_KFeO * (
                                M_Mg * M_O * (
                                M_FeO * (M_MgSiO3 * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Si * (M_Mg * (
                                M_FeO * (M_MgSiO3 * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                M_FeO * (-9.0 * M_MgSiO3 - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                -9.0 * M_FeO + 2.0 * M_SiO2 + 10.0 * M_m))) + M_MgSiO3 * M_O * (M_FeO * (
                                -3.0 * M_SiO2 - 6.0 * M_m) + M_FeSiO3 * (6.0 * M_SiO2 - 15.0 * M_m))) + M_c * (M_Mg * (
                                M_FeO * (M_MgSiO3 * (M_SiO2 - M_m) + M_O * (
                                M_FeSiO3 + M_MgSiO3 + M_SiO2) - M_SiO2 * M_m) + M_FeSiO3 * (
                                M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * M_O * (
                                                                                                               M_FeO * M_SiO2 + M_FeSiO3 * M_m) + M_Si * (
                                                                                                               M_Mg * (
                                                                                                               M_FeO * (
                                                                                                               4.0 * M_MgSiO3 + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                               4.0 * M_FeO - M_SiO2 - 3.0 * M_m) + M_O * (
                                                                                                               M_FeO + M_FeSiO3)) + M_MgSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               2.0 * M_SiO2 + 2.0 * M_m) + M_FeSiO3 * (
                                                                                                               -2.0 * M_SiO2 + 6.0 * M_m)))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                        M_O * (M_Fe * M_FeO * M_MgSiO3 * (4.0 * M_SiO2 - 4.0 * M_m) + M_Mg * (
                        4.0 * M_Fe * M_SiO2 * M_m + M_FeO * M_MgSiO3 * (4.0 * M_SiO2 - 4.0 * M_m))) + M_Si * (M_Mg * (
                        M_Fe * (M_O * (2.0 * M_SiO2 + 10.0 * M_m) + M_SiO2 * M_m) + M_FeO * (
                        M_MgSiO3 * (M_SiO2 - M_m) + M_O * (9.0 * M_MgSiO3 + 3.0 * M_SiO2 + 6.0 * M_m))) + M_MgSiO3 * (
                                                                                                              M_Fe * (
                                                                                                              M_FeO * (
                                                                                                              M_SiO2 - M_m) + M_O * (
                                                                                                              9.0 * M_FeO + 6.0 * M_SiO2 - 15.0 * M_m)) + M_FeO * M_O * (
                                                                                                              9.0 * M_SiO2 - 9.0 * M_m))) + M_c * (
                        M_Mg * (-M_Fe * M_SiO2 * M_m + M_FeO * (
                        M_MgSiO3 * (-M_SiO2 + M_m) + M_O * (-M_MgSiO3 - M_SiO2))) + M_MgSiO3 * (
                        M_Fe * (M_FeO * (-M_SiO2 + M_m) + M_O * (-M_FeO + M_m)) + M_FeO * M_O * (
                        -M_SiO2 + M_m)) + M_Si * (M_Mg * (M_Fe * (M_O - M_SiO2 - 3.0 * M_m) + M_FeO * (
                        -4.0 * M_MgSiO3 - 2.0 * M_SiO2 - 2.0 * M_m)) + M_MgSiO3 * (
                                                  M_Fe * (-4.0 * M_FeO - 2.0 * M_SiO2 + 6.0 * M_m) + M_FeO * (
                                                  -4.0 * M_SiO2 + 4.0 * M_m))))) + M_Mg * (M_O * (M_Fe * (M_FeSiO3 * (
        M_SiO2 * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_MgSiO3 * (M_SiO2 * (
        -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                              -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                          -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_FeO * M_SiO2 * M_m * (
                                                                                                  -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_FeSiO3 * (
                                                                                                  M_FeO * (M_SiO2 * (
                                                                                                  4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                                           -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                                                  -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                                                                                                  M_FeO * (M_SiO2 * (
                                                                                                  -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                                                           4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                  M_SiO2 * (
                                                                                                  -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                                                  -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)))) + dKMgO_KMgO * (
                                                                                           M_Fe * M_O * (M_FeO * (
                                                                                           M_MgSiO3 * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                         M_FeO * (
                                                                                                         4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Si * (
                                                                                           M_Fe * (M_FeO * (M_MgSiO3 * (
                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                   M_FeO * (
                                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                                                                   M_FeO * (
                                                                                                   M_SiO2 - 4.0 * M_m) + M_FeSiO3 * (
                                                                                                   9.0 * M_FeO + 4.0 * M_SiO2 - 25.0 * M_m) + M_MgSiO3 * (
                                                                                                   9.0 * M_FeO + 6.0 * M_SiO2 - 15.0 * M_m) - 9.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           -9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           9.0 * M_SiO2 - 9.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           6.0 * M_SiO2 - 15.0 * M_m) + M_FeSiO3 * (
                                                                                           6.0 * M_SiO2 - 15.0 * M_m)))) + M_c * (
                                                                                           M_Fe * (M_FeO * (M_MgSiO3 * (
                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                   M_FeO * (
                                                                                                   -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                                                                   M_FeSiO3 * (
                                                                                                   -M_FeO + M_m) + M_MgSiO3 * (
                                                                                                   -M_FeO + M_m) + M_SiO2 * (
                                                                                                   -M_FeO + M_m))) + M_O * (
                                                                                           M_FeSiO3 * (M_FeO * (
                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_m * (
                                                                                           M_FeO * M_SiO2 + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_Si * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           -M_SiO2 + M_m) + M_FeSiO3 * (
                                                                                                   -4.0 * M_FeO - M_SiO2 + 9.0 * M_m) + M_MgSiO3 * (
                                                                                                   -4.0 * M_FeO - 2.0 * M_SiO2 + 6.0 * M_m) + M_O * (
                                                                                                   -M_FeO - M_FeSiO3 - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + 4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           -2.0 * M_SiO2 + 6.0 * M_m) + M_FeSiO3 * (
                                                                                           -2.0 * M_SiO2 + 6.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           -M_SiO2 + M_m) + M_FeSiO3 * (
                                                                                           -M_SiO2 + M_m)))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                        M_O * (M_Fe * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Mg * (
                               M_FeSiO3 * (M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                               4.0 * M_Fe + 4.0 * M_FeO))) + M_Si * (M_Fe * (
                        M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                        M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                        -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_Mg * (M_Fe * (
                        M_O * (2.0 * M_SiO2 + 10.0 * M_m) + M_SiO2 * M_m) + M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (
                        -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (M_FeO * (2.0 * M_SiO2 + 10.0 * M_m) + M_FeSiO3 * (
                        -9.0 * M_FeO + 2.0 * M_SiO2 + 10.0 * M_m))) + M_O * (9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                        M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_c * (M_Fe * (
                        -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                        M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m))) + M_Mg * (M_FeSiO3 * (
                        M_FeO * (M_O + M_SiO2 - M_m) - M_SiO2 * M_m) + M_SiO2 * M_m * (-M_Fe - M_FeO)) + M_O * (
                                                                                             -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                             M_FeO * (
                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m)) + M_Si * (
                                                                                             M_Fe * (M_FeO * (
                                                                                             M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                     4.0 * M_FeO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                     M_FeO + M_FeSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) - 4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                             M_FeO * (
                                                                                             4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_Mg * (
                                                                                             M_Fe * (
                                                                                             M_O - M_SiO2 - 3.0 * M_m) + M_FeO * (
                                                                                             -M_SiO2 - 3.0 * M_m) + M_FeSiO3 * (
                                                                                             4.0 * M_FeO - M_SiO2 - 3.0 * M_m) + M_O * (
                                                                                             M_FeO + M_FeSiO3)) + M_O * (
                                                                                             M_FeO * (
                                                                                             M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                             M_SiO2 - M_m))))) + M_Si * (
                        M_Fe * (M_FeO * M_SiO2 * M_m * (-dM_MgO_er - dM_MgSiO3_er) + M_FeSiO3 * (M_FeO * (
                        M_SiO2 * (dM_MgO_er + dM_MgSiO3_er) + M_m * (-dM_MgO_er - dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                                                 -dM_MgO_er - dM_MgSiO3_er)) + M_MgSiO3 * (
                                M_FeO * (M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                M_SiO2 * (-1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er))) + M_O * (M_FeO * (
                        M_SiO2 * (dM_MgO_er + dM_MgSiO3_er) + M_m * (
                        -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                        9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                              4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                                                                              -25.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                  -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_m * (
                                                                                                  6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er + 9.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                  -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er))) + M_Mg * (
                        M_Fe * (M_FeSiO3 * (
                        M_SiO2 * (-1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                        -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                M_SiO2 * (-1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_O * (M_FeSiO3 * (
                        -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                                                 -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                 -2 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                 -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_FeO * M_SiO2 * M_m * (
                        -1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_FeSiO3 * (M_FeO * (
                        M_SiO2 * (1.0 * dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                        -1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                             -1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                        M_FeO * (
                        M_SiO2 * (-dM_FeO_er - 2.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                        1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                        M_SiO2 * (-1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                        -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er))) + M_O * (M_FeO * (M_SiO2 * (
                        -3.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                   -6.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                          M_FeO * (
                                                                                          -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                                          -3.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                          6.0 * dM_FeO_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                          M_FeO * (
                                                                                          -15.0 * dM_FeO_er - 24.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                          -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)))) + M_O * (
                        M_FeO * M_SiO2 * M_m * (-9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_FeSiO3 * (M_FeO * (
                        M_SiO2 * (9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_m * (
                        -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                                                     -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                        M_FeO * (M_SiO2 * (
                        -9.0 * dM_FeO_er - 18.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_m * (
                                 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er + 9.0 * dM_SiO2_er)) + M_FeSiO3 * (
                        M_SiO2 * (-9.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_m * (
                        -9.0 * dM_FeO_er - 9.0 * dM_MgO_er + 9.0 * dM_SiO2_er)))) + dKSiO2_KSiO2 * (M_O * (
                        M_Fe * M_MgSiO3 * (
                        M_FeO * (2.0 * M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (-4.0 * M_SiO2 + 10.0 * M_m)) + M_Mg * (M_Fe * (
                        M_FeSiO3 * (-4.0 * M_SiO2 + 10.0 * M_m) + M_MgSiO3 * (
                        -4.0 * M_SiO2 + 10.0 * M_m) + 6.0 * M_SiO2 * M_m) + 6.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                                M_FeO * (
                                                                                                                -6.0 * M_SiO2 + 6.0 * M_m) + 6.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                                M_FeO * (
                                                                                                                -4.0 * M_SiO2 + 10.0 * M_m) + M_FeSiO3 * (
                                                                                                                -4.0 * M_SiO2 + 10.0 * M_m)))) + M_c * (
                                                                                                    M_Mg * (M_Fe * (
                                                                                                    M_FeSiO3 * (
                                                                                                    M_SiO2 - 3.0 * M_m) + M_MgSiO3 * (
                                                                                                    M_SiO2 - 3.0 * M_m) + M_O * (
                                                                                                    M_FeSiO3 + M_MgSiO3 + M_SiO2) - 2.0 * M_SiO2 * M_m) - 2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                            M_FeO * (
                                                                                                            2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                            M_FeO * (
                                                                                                            M_SiO2 - 3.0 * M_m) + M_FeSiO3 * (
                                                                                                            M_SiO2 - 3.0 * M_m)) + M_O * (
                                                                                                            M_MgSiO3 * (
                                                                                                            M_FeO + M_FeSiO3) + M_SiO2 * (
                                                                                                            M_FeO + M_FeSiO3))) + M_MgSiO3 * (
                                                                                                    M_Fe * (M_FeO * (
                                                                                                    -M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                            M_SiO2 - 3.0 * M_m) + M_O * (
                                                                                                            M_FeO + M_FeSiO3 + M_SiO2 - M_m)) + M_O * (
                                                                                                    M_FeO * (
                                                                                                    M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)))))) + M_c * (
                        M_Fe * (M_FeO * M_SiO2 * M_m * (dM_MgO_er + dM_MgSiO3_er) + M_FeSiO3 * (M_FeO * (
                        M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                                                dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (
                                M_FeO * (M_SiO2 * (
                                dM_FeO_er + 2.0 * dM_FeSiO3_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                         -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er)) + M_FeSiO3 * (
                                M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er))) + M_O * (M_FeSiO3 * (
                        M_FeO * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
                        dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er) + M_FeSiO3 * (
                                                                                                              dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_SiO2 * (
                                                                                                              dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                              -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er)) + M_SiO2 * (
                                                                                     M_FeO * (
                                                                                     -dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                                     dM_MgO_er + dM_MgSiO3_er)))) + M_Mg * (
                        M_Fe * (M_FeSiO3 * (M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_O * (
                                M_FeSiO3 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_MgSiO3 * (
                                dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_SiO2 * (
                                dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_FeO * M_SiO2 * M_m * (
                        dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_FeSiO3 * (M_FeO * (
                        M_SiO2 * (-dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                        dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                      dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                        M_FeO * (M_SiO2 * (dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        -1.0 * dM_FeSiO3_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                        M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er))) + M_O * (
                        M_FeO * M_SiO2 * (dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
                        M_FeO * (dM_FeO_er + dM_FeSiO3_er) + M_SiO2 * (
                        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                        M_FeO * (dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
                        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)))) + M_O * (
                        M_FeO * M_SiO2 * M_m * (dM_MgO_er + dM_MgSiO3_er) + M_FeSiO3 * (M_FeO * (
                        M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                                        dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (
                        M_FeO * (M_SiO2 * (dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er)) + M_FeSiO3 * (
                        M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er + dM_MgO_er - dM_SiO2_er)))) + M_Si * (M_Fe * (
                        M_FeO * (M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (
                        M_FeO * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (
                        9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
                        2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                             2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                             2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                             -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_O * (
                        M_FeO * (-dM_MgO_er - dM_MgSiO3_er) + M_FeSiO3 * (-dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                        -dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_SiO2 * M_m * (
                        4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_FeO * M_SiO2 * M_m * (
                                                                         4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_FeSiO3 * (
                                                                         M_FeO * (M_SiO2 * (
                                                                         -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                  4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_SiO2 * M_m * (
                                                                         4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_Mg * (
                                                                         M_Fe * (M_FeSiO3 * (
                                                                         2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                                 2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                 dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                 dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_FeO * (
                                                                         M_SiO2 * (
                                                                         2 * dM_FeO_er + 4.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                         2.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                         M_FeO * (
                                                                         4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                         2.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                         -2.0 * dM_FeO_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                         M_FeO * (
                                                                         6.0 * dM_FeO_er + 10.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                         2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_O * (
                                                                         M_FeO * (
                                                                         dM_FeO_er + dM_FeSiO3_er) + M_FeSiO3 * (
                                                                         dM_FeO_er + dM_FeSiO3_er))) + M_MgSiO3 * (
                                                                         M_FeO * (M_SiO2 * (
                                                                         4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                  -4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                         M_SiO2 * (
                                                                         4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                         4.0 * dM_FeO_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er))) + M_O * (
                                                                         M_FeO * (
                                                                         M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                         dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                         M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                         dM_MgO_er + dM_MgSiO3_er)))))) / (M_O * (
        M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                     M_FeO * (
                                                                                     -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                     -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                     4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                             4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                             M_FeO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                             M_FeO * (M_MgO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                             M_FeO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
                                                                                                           M_Fe * (
                                                                                                           M_MgO * (
                                                                                                           M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + M_m) + M_MgO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                                           -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
                                                                                                           M_Fe * (
                                                                                                           M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + M_m) + M_MgO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + M_m) + M_MgO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                                           -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                                                                                                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                           M_FeO + M_MgO)) + M_MgO * (
                                                                                                           M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 + M_m) + M_MgO * (
                                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                                           -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
                                                                                                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
                                                                                                           M_Fe * (
                                                                                                           M_MgO * (
                                                                                                           -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           M_FeSiO3 * (
                                                                                                           M_FeO - M_m) + M_SiO2 * (
                                                                                                           M_FeO - M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                                           M_FeO + M_MgO - M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (
                                                                                                           M_Fe * (
                                                                                                           M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                                                                           M_FeSiO3 * (
                                                                                                           M_FeO + M_MgO - M_m) + M_MgSiO3 * (
                                                                                                           M_FeO + M_MgO - M_m) + M_SiO2 * (
                                                                                                           M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (
                                                                                                           -M_FeO - M_MgO)) + M_MgO * (
                                                                                                           -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                                           M_FeO * M_SiO2 * (
                                                                                                           M_MgO - M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                                                                           M_MgO - M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO - M_m) + M_FeSiO3 * (
                                                                                                           M_FeO + M_MgO - M_m)))) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (
                                                                                                           M_Fe * (
                                                                                                           M_MgO * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                           4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                           M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                                           M_Fe * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                           M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                           M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                           M_SiO2 - M_m))))))

    def dM_SiO2_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_SiO2 * (M_Fe * (M_O * (M_MgO * (
        M_FeO * M_m * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_FeSiO3 * (
        M_FeO * (-4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
        4.0 * dM_FeO_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er))) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
        -4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                 -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                 4.0 * dM_FeO_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)))) + dKFeO_KFeO * (
                                 M_Mg * M_O * M_m * (-4.0 * M_FeO * M_MgSiO3 + 4.0 * M_FeSiO3 * M_MgO) + M_Si * (
                                 M_Mg * (M_O * (M_FeO * (-6.0 * M_MgSiO3 + 6.0 * M_m) + M_FeSiO3 * (
                                 -9.0 * M_FeO - 3.0 * M_MgO + 15.0 * M_m)) + M_m * (
                                         -M_FeO * M_MgSiO3 + M_FeSiO3 * M_MgO)) + M_O * (
                                 M_MgO * (6.0 * M_FeO * M_m + M_FeSiO3 * (-9.0 * M_FeO + 15.0 * M_m)) + M_MgSiO3 * (
                                 M_FeO * (-9.0 * M_MgO + 6.0 * M_m) + M_FeSiO3 * (
                                 -9.0 * M_FeO - 9.0 * M_MgO + 15.0 * M_m)))) + M_c * (M_Mg * (
                                 M_FeO * M_MgSiO3 * M_m + M_FeSiO3 * (
                                 -M_MgO * M_m + M_O * (M_FeO + M_MgO - M_m))) + M_O * (M_FeSiO3 * M_MgO * (
                                 M_FeO - M_m) + M_MgSiO3 * (M_FeO * M_MgO + M_FeSiO3 * (
                                 M_FeO + M_MgO - M_m))) + M_Si * (M_Mg * (
                                 M_FeO * (2.0 * M_MgSiO3 - 2.0 * M_m) + M_FeSiO3 * (
                                 4.0 * M_FeO + 2.0 * M_MgO - 6.0 * M_m) + M_O * (M_FeO + M_FeSiO3)) + M_MgO * (
                                                                  -2.0 * M_FeO * M_m + M_FeSiO3 * (
                                                                  4.0 * M_FeO - 6.0 * M_m)) + M_MgSiO3 * (
                                                                  M_FeO * (4.0 * M_MgO - 2.0 * M_m) + M_FeSiO3 * (
                                                                  4.0 * M_FeO + 4.0 * M_MgO - 6.0 * M_m)) + M_O * (
                                                                  M_MgO * (M_FeO + M_FeSiO3) + M_MgSiO3 * (
                                                                  M_FeO + M_FeSiO3)))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                         M_O * M_m * (M_Fe * M_FeO * (4.0 * M_MgO + 4.0 * M_MgSiO3) + M_Mg * (
                         M_Fe * (4.0 * M_FeO + 4.0 * M_MgO) + M_FeO * (4.0 * M_MgO + 4.0 * M_MgSiO3))) + M_Si * (
                         M_Fe * (M_FeO * M_m * (M_MgO + M_MgSiO3) + M_O * (
                         M_MgO * (-3.0 * M_FeO + 15.0 * M_m) + M_MgSiO3 * (
                         -3.0 * M_FeO - 9.0 * M_MgO + 15.0 * M_m))) + M_FeO * M_O * M_m * (
                         9.0 * M_MgO + 9.0 * M_MgSiO3) + M_Mg * (
                         M_Fe * (M_O * (-3.0 * M_FeO - 3.0 * M_MgO + 15.0 * M_m) + M_m * (M_FeO + M_MgO)) + M_FeO * (
                         M_O * (-3.0 * M_MgO + 6.0 * M_MgSiO3 + 9.0 * M_m) + M_m * (M_MgO + M_MgSiO3)))) + M_c * (
                         M_Fe * (M_FeO * M_m * (-M_MgO - M_MgSiO3) + M_O * (
                         M_MgO * (M_FeO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m))) + M_FeO * M_O * M_m * (
                         -M_MgO - M_MgSiO3) + M_Mg * (
                         M_Fe * (M_O * (M_FeO + M_MgO - M_m) + M_m * (-M_FeO - M_MgO)) + M_FeO * (
                         M_O * (M_MgO - M_m) + M_m * (-M_MgO - M_MgSiO3))) + M_Si * (M_Fe * (
                         M_MgO * (2.0 * M_FeO - 6.0 * M_m) + M_MgSiO3 * (
                         2.0 * M_FeO + 4.0 * M_MgO - 6.0 * M_m) + M_O * (M_MgO + M_MgSiO3)) + M_FeO * M_m * (
                                                                                     -4.0 * M_MgO - 4.0 * M_MgSiO3) + M_Mg * (
                                                                                     M_Fe * (
                                                                                     2.0 * M_FeO + 2.0 * M_MgO + M_O - 6.0 * M_m) + M_FeO * (
                                                                                     2.0 * M_MgO - 2.0 * M_MgSiO3 - 4.0 * M_m))))) + M_Mg * (
                         M_O * (M_Fe * (M_FeSiO3 * (M_FeO * (
                         -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                    -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                                    4.0 * dM_FeO_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                        M_FeO * (
                                        -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                        -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                        4.0 * dM_FeO_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_m * (M_FeO * (
                         -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_MgO * (
                                                                                                        -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er))) + M_MgO * (
                                M_FeO * M_m * (
                                -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_FeSiO3 * (M_FeO * (
                                -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                                           4.0 * dM_FeO_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                M_FeO * (M_MgO * (
                                -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                         -4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                M_FeO * (
                                -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                4.0 * dM_FeO_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)))) + dKMgO_KMgO * (
                         M_Fe * M_O * M_m * (4.0 * M_FeO * M_MgSiO3 - 4.0 * M_FeSiO3 * M_MgO) + M_Si * (M_Fe * (M_O * (
                         M_MgO * (-6.0 * M_FeSiO3 + 6.0 * M_m) + M_MgSiO3 * (
                         -3.0 * M_FeO - 9.0 * M_MgO + 15.0 * M_m)) + M_m * (
                                                                                                                M_FeO * M_MgSiO3 - M_FeSiO3 * M_MgO)) + M_O * (
                                                                                                        M_MgO * (
                                                                                                        6.0 * M_FeO * M_m + M_FeSiO3 * (
                                                                                                        -9.0 * M_FeO + 6.0 * M_m)) + M_MgSiO3 * (
                                                                                                        M_FeO * (
                                                                                                        -9.0 * M_MgO + 15.0 * M_m) + M_FeSiO3 * (
                                                                                                        -9.0 * M_FeO - 9.0 * M_MgO + 15.0 * M_m)))) + M_c * (
                         M_Fe * (
                         M_FeSiO3 * M_MgO * M_m + M_MgSiO3 * (-M_FeO * M_m + M_O * (M_FeO + M_MgO - M_m))) + M_O * (
                         M_FeO * M_FeSiO3 * M_MgO + M_MgSiO3 * (
                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (M_FeO + M_MgO - M_m))) + M_Si * (M_Fe * (
                         M_MgO * (2.0 * M_FeSiO3 - 2.0 * M_m) + M_MgSiO3 * (
                         2.0 * M_FeO + 4.0 * M_MgO - 6.0 * M_m) + M_O * (M_MgO + M_MgSiO3)) + M_MgO * (
                                                                                              -2.0 * M_FeO * M_m + M_FeSiO3 * (
                                                                                              4.0 * M_FeO - 2.0 * M_m)) + M_MgSiO3 * (
                                                                                              M_FeO * (
                                                                                              4.0 * M_MgO - 6.0 * M_m) + M_FeSiO3 * (
                                                                                              4.0 * M_FeO + 4.0 * M_MgO - 6.0 * M_m)) + M_O * (
                                                                                              M_MgO * (
                                                                                              M_FeO + M_FeSiO3) + M_MgSiO3 * (
                                                                                              M_FeO + M_FeSiO3)))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                         M_O * M_m * (M_Fe * M_MgO * (4.0 * M_FeO + 4.0 * M_FeSiO3) + M_Mg * (
                         M_Fe * (4.0 * M_FeO + 4.0 * M_MgO) + M_MgO * (4.0 * M_FeO + 4.0 * M_FeSiO3))) + M_Si * (
                         M_Mg * (M_Fe * (
                         M_O * (-3.0 * M_FeO - 3.0 * M_MgO + 15.0 * M_m) + M_m * (M_FeO + M_MgO)) + M_MgO * M_m * (
                                 M_FeO + M_FeSiO3) + M_O * (M_FeO * (-3.0 * M_MgO + 15.0 * M_m) + M_FeSiO3 * (
                         -9.0 * M_FeO - 3.0 * M_MgO + 15.0 * M_m))) + M_MgO * (M_Fe * (
                         M_O * (-3.0 * M_FeO + 6.0 * M_FeSiO3 + 9.0 * M_m) + M_m * (M_FeO + M_FeSiO3)) + M_O * M_m * (
                                                                               9.0 * M_FeO + 9.0 * M_FeSiO3))) + M_c * (
                         M_Mg * (M_Fe * (M_O * (M_FeO + M_MgO - M_m) + M_m * (-M_FeO - M_MgO)) + M_MgO * M_m * (
                         -M_FeO - M_FeSiO3) + M_O * (
                                 M_FeO * (M_MgO - M_m) + M_FeSiO3 * (M_FeO + M_MgO - M_m))) + M_MgO * (
                         M_Fe * (M_O * (M_FeO - M_m) + M_m * (-M_FeO - M_FeSiO3)) + M_O * M_m * (
                         -M_FeO - M_FeSiO3)) + M_Si * (M_Mg * (
                         M_Fe * (2.0 * M_FeO + 2.0 * M_MgO + M_O - 6.0 * M_m) + M_FeO * (
                         2.0 * M_MgO - 6.0 * M_m) + M_FeSiO3 * (4.0 * M_FeO + 2.0 * M_MgO - 6.0 * M_m) + M_O * (
                         M_FeO + M_FeSiO3)) + M_MgO * (M_Fe * (2.0 * M_FeO - 2.0 * M_FeSiO3 - 4.0 * M_m) + M_m * (
                         -4.0 * M_FeO - 4.0 * M_FeSiO3))))) + M_Si * (M_Fe * (M_MgO * (
        M_FeO * M_m * (-1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_FeSiO3 * (
        M_FeO * (-1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
        1.0 * dM_FeO_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er))) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
        -1.0 * dM_FeSiO3_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                 -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                 1.0 * dM_FeO_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er))) + M_O * (
                                                                              M_MgO * (M_FeO * (
                                                                              2 * dM_FeO_er + 5.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                       -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 16.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
                                                                                       -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                              M_FeO * (
                                                                              2 * dM_FeO_er + 5.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                              -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_MgO * (
                                                                              -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_m * (
                                                                              -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er - 9.0 * dM_SiO2_er)))) + M_Mg * (
                                                                      M_Fe * (M_FeSiO3 * (M_FeO * (
                                                                      -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                          -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                          1.0 * dM_FeO_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                              M_FeO * (
                                                                              -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                              -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                              1.0 * dM_FeO_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er)) + M_O * (
                                                                              M_FeO * (
                                                                              2 * dM_FeO_er + 5.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                              -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_MgO * (
                                                                              2 * dM_FeO_er + 5.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                              -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
                                                                              -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_m * (
                                                                              M_FeO * (
                                                                              -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_MgO * (
                                                                              -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er))) + M_MgO * (
                                                                      M_FeO * M_m * (
                                                                      -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                      M_FeO * (
                                                                      -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                      1.0 * dM_FeO_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                                                      M_FeO * (M_MgO * (
                                                                      -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                               -1.0 * dM_FeSiO3_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                      M_FeO * (
                                                                      -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                      -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                      1.0 * dM_FeO_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er))) + M_O * (
                                                                      M_FeO * (M_MgO * (
                                                                      2 * dM_FeO_er + 5.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                                                                               -9.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                      M_FeO * (
                                                                      -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er) + M_MgO * (
                                                                      -1.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                                                                      9.0 * dM_FeO_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                      M_FeO * (
                                                                      -10.0 * dM_FeO_er - 16.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                      -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)))) + M_O * (
                                                                      M_MgO * (M_FeO * M_m * (
                                                                      -9.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                               M_FeO * (
                                                                               -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_m * (
                                                                               9.0 * dM_FeO_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                                                      M_FeO * (M_MgO * (
                                                                      -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_m * (
                                                                               -9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er - 9.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                      M_FeO * (
                                                                      -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_MgO * (
                                                                      -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_m * (
                                                                      9.0 * dM_FeO_er + 9.0 * dM_MgO_er - 9.0 * dM_SiO2_er)))) + dKSiO2_KSiO2 * (
                                                                      M_O * (M_Fe * (M_MgO * (
                                                                      -4.0 * M_FeO * M_m + M_FeSiO3 * (
                                                                      6.0 * M_FeO - 10.0 * M_m)) + M_MgSiO3 * (M_FeO * (
                                                                      6.0 * M_MgO - 4.0 * M_m) + M_FeSiO3 * (
                                                                                                               6.0 * M_FeO + 6.0 * M_MgO - 10.0 * M_m))) + M_Mg * (
                                                                             M_Fe * (M_FeSiO3 * (
                                                                             6.0 * M_FeO + 6.0 * M_MgO - 10.0 * M_m) + M_MgSiO3 * (
                                                                                     6.0 * M_FeO + 6.0 * M_MgO - 10.0 * M_m) + M_m * (
                                                                                     -4.0 * M_FeO - 4.0 * M_MgO)) + M_MgO * (
                                                                             -4.0 * M_FeO * M_m + M_FeSiO3 * (
                                                                             6.0 * M_FeO - 4.0 * M_m)) + M_MgSiO3 * (
                                                                             M_FeO * (
                                                                             6.0 * M_MgO - 10.0 * M_m) + M_FeSiO3 * (
                                                                             6.0 * M_FeO + 6.0 * M_MgO - 10.0 * M_m)))) + M_c * (
                                                                      M_Fe * (M_MgO * (M_FeO * M_m + M_FeSiO3 * (
                                                                      -2.0 * M_FeO + 3.0 * M_m)) + M_MgSiO3 * (M_FeO * (
                                                                      -2.0 * M_MgO + M_m) + M_FeSiO3 * (
                                                                                                               -2.0 * M_FeO - 2.0 * M_MgO + 3.0 * M_m)) + M_O * (
                                                                              M_MgO * (-M_FeO + M_m) + M_MgSiO3 * (
                                                                              -M_FeO + M_m))) + M_Mg * (M_Fe * (
                                                                      M_FeSiO3 * (
                                                                      -2.0 * M_FeO - 2.0 * M_MgO + 3.0 * M_m) + M_MgSiO3 * (
                                                                      -2.0 * M_FeO - 2.0 * M_MgO + 3.0 * M_m) + M_O * (
                                                                      -M_FeO - M_MgO + M_m) + M_m * (
                                                                      M_FeO + M_MgO)) + M_MgO * (
                                                                                                        M_FeO * M_m + M_FeSiO3 * (
                                                                                                        -2.0 * M_FeO + M_m)) + M_MgSiO3 * (
                                                                                                        M_FeO * (
                                                                                                        -2.0 * M_MgO + 3.0 * M_m) + M_FeSiO3 * (
                                                                                                        -2.0 * M_FeO - 2.0 * M_MgO + 3.0 * M_m)) + M_O * (
                                                                                                        M_FeO * (
                                                                                                        -M_MgO + M_m) + M_FeSiO3 * (
                                                                                                        -M_MgO + M_m))) + M_O * M_m * (
                                                                      M_MgO * (M_FeO + M_FeSiO3) + M_MgSiO3 * (
                                                                      M_FeO + M_FeSiO3))))) + M_c * (M_Fe * (M_MgO * (
        M_FeO * M_m * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
        M_FeO * (dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
        -dM_FeO_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er))) + M_MgSiO3 * (M_FeO * (
        M_MgO * (dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
        dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                              dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                                                              -dM_FeO_er - dM_MgO_er + 1.0 * dM_SiO2_er))) + M_O * (
                                                                                                             M_MgO * (
                                                                                                             M_FeO * (
                                                                                                             -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                             dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er) + M_MgO * (
                                                                                                             dM_MgO_er + dM_MgSiO3_er) + M_m * (
                                                                                                             dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)))) + M_Mg * (
                                                                                                     M_Fe * (
                                                                                                     M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                                     -dM_FeO_er - dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                                     -dM_FeO_er - dM_MgO_er + 1.0 * dM_SiO2_er)) + M_O * (
                                                                                                     M_FeO * (
                                                                                                     -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                                                                                                     -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                     dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_m * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_MgO * (
                                                                                                     dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er))) + M_MgO * (
                                                                                                     M_FeO * M_m * (
                                                                                                     dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                                     -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er))) + M_MgSiO3 * (
                                                                                                     M_FeO * (M_MgO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                                              1.0 * dM_FeSiO3_er - dM_MgO_er + 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                     dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                                     -dM_FeO_er - dM_MgO_er + 1.0 * dM_SiO2_er))) + M_O * (
                                                                                                     M_FeO * (M_MgO * (
                                                                                                     -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                              dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er) + M_MgO * (
                                                                                                     dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                     -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er)))) + M_O * (
                                                                                                     M_MgO * (
                                                                                                     M_FeO * M_m * (
                                                                                                     dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_m * (
                                                                                                     -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er))) + M_MgSiO3 * (
                                                                                                     M_FeO * (M_MgO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_m * (
                                                                                                              dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_MgO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_m * (
                                                                                                     -dM_FeO_er - dM_MgO_er + dM_SiO2_er)))) + M_Si * (
                                                                                                     M_Fe * (M_MgO * (
                                                                                                     M_FeO * (
                                                                                                     -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                     dM_FeO_er + 3.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                                     2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             -dM_FeO_er - 3.0 * dM_FeSiO3_er + 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                             dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_MgO * (
                                                                                                             4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                                             2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_O * (
                                                                                                             M_MgO * (
                                                                                                             dM_MgO_er + dM_MgSiO3_er) + M_MgSiO3 * (
                                                                                                             dM_MgO_er + dM_MgSiO3_er))) + M_Mg * (
                                                                                                     M_Fe * (M_FeO * (
                                                                                                     -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                             dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_MgO * (
                                                                                                             -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                                                             dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                                             2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_FeO * (
                                                                                                     M_MgO * (
                                                                                                     -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                                     4.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_MgO * (
                                                                                                     1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                                     -4.0 * dM_FeO_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     3.0 * dM_FeO_er + 5.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                     dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_O * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er) + M_FeSiO3 * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er))) + M_MgO * (
                                                                                                     M_FeO * M_m * (
                                                                                                     4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                                     -4.0 * dM_FeO_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                                                                                     M_FeO * (M_MgO * (
                                                                                                     4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                                              4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                     4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (
                                                                                                     -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er))) + M_O * (
                                                                                                     M_MgO * (M_FeO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_FeSiO3 * (
                                                                                                              dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_FeSiO3 * (
                                                                                                     dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er)))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_FeSiO3_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_FeSiO3 * (M_Fe * (M_O * (M_MgO * (M_FeO * (M_SiO2 * (
        4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                            -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                   -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (
        M_SiO2 * (4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
        4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (-4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                         -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er))) + dKFeO_KFeO * (
                                   M_Mg * M_O * (-4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                   M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                   4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Si * (M_Mg * (
                                   -M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                   M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                   M_FeO * (3.0 * M_SiO2 + 6.0 * M_m) + M_MgO * (M_SiO2 - 4.0 * M_m) + M_MgSiO3 * (
                                   9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m) - 9.0 * M_SiO2 * M_m)) + M_O * (
                                                                                              M_MgO * (M_FeO * (
                                                                                              3.0 * M_SiO2 + 6.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                              M_FeO * (
                                                                                              3.0 * M_SiO2 + 6.0 * M_m) + M_MgO * (
                                                                                              9.0 * M_SiO2 - 9.0 * M_m) - 9.0 * M_SiO2 * M_m))) + M_c * (
                                   M_Mg * (M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                   M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                           M_MgSiO3 * (-M_FeO - M_MgO + M_m) + M_SiO2 * (
                                           -M_FeO - M_MgO + M_m))) + M_O * (
                                   M_MgO * M_SiO2 * (-M_FeO + M_m) + M_MgSiO3 * (
                                   M_MgO * (-M_SiO2 + M_m) + M_SiO2 * (-M_FeO + M_m))) + M_Si * (M_Mg * (
                                   M_FeO * (-2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (-M_SiO2 + M_m) + M_MgSiO3 * (
                                   -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_O * (
                                   -M_MgO - M_MgSiO3 - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_MgO * (M_FeO * (
                                   -2.0 * M_SiO2 - 2.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
                                   -2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (
                                                                                                  -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                 M_MgO * (
                                                                                                 -M_SiO2 + M_m) + M_MgSiO3 * (
                                                                                                 -M_SiO2 + M_m)))))) + M_Mg * (
                           M_O * (M_Fe * (M_FeO * (M_SiO2 * (
                           4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                   -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_MgO * (
                                          M_SiO2 * (
                                          4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                          -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                          -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgO * (
                                  M_FeO * (M_SiO2 * (
                                  4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                           -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                  -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (M_SiO2 * (
                           4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                                 -4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_MgO * (
                                                                                        M_SiO2 * (
                                                                                        4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                        -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                        -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er))) + dKMgO_KMgO * (
                           M_Fe * M_O * (4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                           M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Si * (M_Fe * (
                           M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                           M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                           M_MgO * (2.0 * M_SiO2 + 10.0 * M_m) + M_MgSiO3 * (
                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m))) + M_FeO * M_O * (M_MgO * (
                           3.0 * M_SiO2 + 6.0 * M_m) + M_MgSiO3 * (-6.0 * M_SiO2 + 15.0 * M_m))) + M_c * (M_Fe * (
                           -M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) + M_O * (
                           M_FeO + M_MgO - M_m) - M_SiO2 * M_m)) + M_FeO * M_O * (
                                                                                                          -M_MgO * M_SiO2 - M_MgSiO3 * M_m) + M_Si * (
                                                                                                          M_Fe * (
                                                                                                          M_MgO * (
                                                                                                          -M_SiO2 - 3.0 * M_m) + M_MgSiO3 * (
                                                                                                          4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                          M_MgO + M_MgSiO3)) + M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          -2.0 * M_SiO2 - 2.0 * M_m) + M_MgSiO3 * (
                                                                                                          2.0 * M_SiO2 - 6.0 * M_m)))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                           M_O * (M_Fe * M_FeO * M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_Mg * (M_Fe * (
                           M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeO * M_MgO * (
                                                                                               -4.0 * M_SiO2 + 4.0 * M_m))) + M_Si * (
                           M_Mg * (M_Fe * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_O * (
                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_SiO2 * M_m) + M_FeO * (
                                   M_MgO * (-M_SiO2 + M_m) + M_O * (
                                   -9.0 * M_MgO - 6.0 * M_SiO2 + 15.0 * M_m))) + M_MgO * (M_Fe * (
                           M_FeO * (-M_SiO2 + M_m) + M_O * (-9.0 * M_FeO - 6.0 * M_SiO2 + 15.0 * M_m)) + M_FeO * M_O * (
                                                                                          -9.0 * M_SiO2 + 9.0 * M_m))) + M_c * (
                           M_Mg * (M_Fe * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) + M_O * (
                           M_FeO + M_MgO - M_m) - M_SiO2 * M_m) + M_FeO * (
                                   M_MgO * (M_SiO2 - M_m) + M_O * (M_MgO - M_m))) + M_MgO * (
                           M_Fe * (M_FeO * (M_SiO2 - M_m) + M_O * (M_FeO - M_m)) + M_FeO * M_O * (
                           M_SiO2 - M_m)) + M_Si * (M_Mg * (
                           M_Fe * (4.0 * M_FeO + 4.0 * M_MgO + M_O + M_SiO2 - 9.0 * M_m) + M_FeO * (
                           4.0 * M_MgO + 2.0 * M_SiO2 - 6.0 * M_m)) + M_MgO * (
                                                    M_Fe * (4.0 * M_FeO + 2.0 * M_SiO2 - 6.0 * M_m) + M_FeO * (
                                                    4.0 * M_SiO2 - 4.0 * M_m))))) + M_Si * (M_Fe * (M_MgO * (M_FeO * (
        M_SiO2 * (dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
        -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                             -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                                    M_FeO * (M_SiO2 * (
                                                                                                    dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                             -1.0 * dM_FeSiO3_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                    M_SiO2 * (
                                                                                                    1.0 * dM_FeO_er + 1.0 * dM_FeSiO3_er) + M_m * (
                                                                                                    -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                    -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_O * (
                                                                                                    M_MgO * (M_FeO * (
                                                                                                    6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                             4.0 * dM_FeO_er + 10.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 12.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_m * (
                                                                                                             -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 15.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                    M_FeO * (
                                                                                                    6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_MgO * (
                                                                                                    -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                    4.0 * dM_FeO_er + 10.0 * dM_FeSiO3_er + 6.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_m * (
                                                                                                    -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er + 15.0 * dM_MgO_er - 15.0 * dM_SiO2_er)))) + M_Mg * (
                                                                                            M_Fe * (M_FeO * (M_SiO2 * (
                                                                                            dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                             -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                    M_SiO2 * (
                                                                                                    dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                    -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_O * (
                                                                                                    M_FeO * (
                                                                                                    6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_MgO * (
                                                                                                    6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                    4.0 * dM_FeO_er + 10.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 10.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_m * (
                                                                                                    -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                    -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                            M_FeO * (M_SiO2 * (
                                                                                            dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                     -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                            -dM_FeO_er - dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                            M_FeO * (M_SiO2 * (
                                                                                            dM_FeO_er + 2.0 * dM_FeSiO3_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                                                                     -1.0 * dM_FeSiO3_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgO * (
                                                                                            M_SiO2 * (
                                                                                            dM_FeO_er + dM_FeSiO3_er) + M_m * (
                                                                                            -dM_FeO_er - dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                            -dM_FeO_er - dM_FeSiO3_er)) + M_O * (
                                                                                            M_FeO * (M_MgO * (
                                                                                            6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                     9.0 * dM_FeO_er + 18.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_m * (
                                                                                                     -9.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_MgO * (
                                                                                            M_SiO2 * (
                                                                                            dM_FeO_er + dM_FeSiO3_er) + M_m * (
                                                                                            -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                            M_FeO * (
                                                                                            15.0 * dM_FeO_er + 24.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_MgO * (
                                                                                            9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                                            4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                            -25.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                            -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er))) + M_O * (
                                                                                            M_MgO * (M_FeO * (M_SiO2 * (
                                                                                            9.0 * dM_FeO_er + 18.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 18.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_m * (
                                                                                                              -9.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                     -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                            M_FeO * (M_SiO2 * (
                                                                                            9.0 * dM_FeO_er + 18.0 * dM_FeSiO3_er + 9.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_m * (
                                                                                                     -9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er - 9.0 * dM_SiO2_er)) + M_MgO * (
                                                                                            M_SiO2 * (
                                                                                            9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er) + M_m * (
                                                                                            -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                            -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er))) + dKSiO2_KSiO2 * (
                                                                                            M_O * (M_Fe * (M_MgO * (
                                                                                            M_FeO * (
                                                                                            -2.0 * M_SiO2 - 4.0 * M_m) + 6.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -2.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                                           -6.0 * M_SiO2 + 6.0 * M_m) + 6.0 * M_SiO2 * M_m)) + M_Mg * (
                                                                                                   M_Fe * (M_FeO * (
                                                                                                   -2.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                                           -2.0 * M_SiO2 - 4.0 * M_m) + 6.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                                   M_MgO * (
                                                                                                   -2.0 * M_SiO2 - 4.0 * M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_SiO2 - 10.0 * M_m)))) + M_c * (
                                                                                            M_Fe * (M_MgO * (M_FeO * (
                                                                                            M_SiO2 + M_m) - 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                    M_FeO * (
                                                                                                    M_SiO2 + M_m) + M_MgO * (
                                                                                                    2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_O * (
                                                                                                    M_MgO * (
                                                                                                    -M_FeO + M_m) + M_MgSiO3 * (
                                                                                                    -M_FeO + M_m))) + M_FeO * M_O * (
                                                                                            M_MgO * (
                                                                                            -M_SiO2 + M_m) + M_MgSiO3 * (
                                                                                            -M_SiO2 + M_m)) + M_Mg * (
                                                                                            M_Fe * (M_FeO * (
                                                                                            M_SiO2 + M_m) + M_MgO * (
                                                                                                    M_SiO2 + M_m) + M_O * (
                                                                                                    -M_FeO - M_MgO + M_m) - 2.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                            M_MgO * (
                                                                                            M_SiO2 + M_m) + M_MgSiO3 * (
                                                                                            -M_SiO2 + 3.0 * M_m) + M_O * (
                                                                                            -M_MgO - M_MgSiO3 - M_SiO2 + M_m)))))) + M_c * (
                           M_Fe * (M_MgO * (M_FeO * (M_SiO2 * (
                           -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                     dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                            dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (
                           M_SiO2 * (-dM_FeO_er - 2.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                           dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                           -dM_FeO_er - 1.0 * dM_FeSiO3_er) + M_m * (dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                           dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_O * (
                                   M_MgO * (M_FeO * (-dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                   dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                   M_FeO * (-dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er) + M_MgO * (
                                   dM_MgO_er + dM_MgSiO3_er) + M_m * (
                                   dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)))) + M_Mg * (M_Fe * (M_FeO * (M_SiO2 * (
                           -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                                        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgO * (
                                                                                               M_SiO2 * (
                                                                                               -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_O * (
                                                                                               M_FeO * (
                                                                                               -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                                                                                               -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                               dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                       M_FeO * (M_SiO2 * (
                                                                                       -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                                dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                       M_FeO * (M_SiO2 * (
                                                                                       -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                1.0 * dM_FeSiO3_er - dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgO * (
                                                                                       M_SiO2 * (
                                                                                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_O * (
                                                                                       M_FeO * (M_MgO * (
                                                                                       -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_SiO2 * (
                                                                                                -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                                                                       M_FeO * (
                                                                                       -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                                                                                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * (
                                                                                       M_MgO * (
                                                                                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)))) + M_O * (
                           M_MgO * (M_FeO * (M_SiO2 * (
                           -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                             dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                    dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (
                           M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                           dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                           -dM_FeO_er - dM_FeSiO3_er) + M_m * (dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                             dM_FeO_er + dM_FeSiO3_er))) + M_Si * (
                           M_Fe * (M_MgO * (M_FeO * (
                           -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                            -dM_FeO_er - 3.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                            3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 6.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                   M_FeO * (
                                   -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_MgO * (
                                   4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                   -dM_FeO_er - 3.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                   3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er + 6.0 * dM_SiO2_er)) + M_O * (
                                   M_MgO * (dM_MgO_er + dM_MgSiO3_er) + M_MgSiO3 * (
                                   dM_MgO_er + dM_MgSiO3_er))) + M_Mg * (M_Fe * (M_FeO * (
                           -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_MgO * (
                                                                                 -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                 -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                 3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_FeO * (
                                                                         M_MgO * (
                                                                         -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                         -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                         4.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_MgO * (
                                                                         M_SiO2 * (-dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                         dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                         M_FeO * (
                                                                         -6.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_MgO * (
                                                                         -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                         -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                         9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er)) + M_O * (
                                                                         M_MgO * (
                                                                         -dM_FeO_er - dM_FeSiO3_er) + M_MgSiO3 * (
                                                                         -dM_FeO_er - dM_FeSiO3_er) + M_SiO2 * (
                                                                         -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                         dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                         4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_MgO * (
                           M_FeO * (M_SiO2 * (
                           -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                    4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                           4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (M_SiO2 * (
                           -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                                         4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_MgO * (
                                                                                M_SiO2 * (
                                                                                -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_O * (
                           M_MgO * (
                           M_SiO2 * (-dM_FeO_er - dM_FeSiO3_er) + M_m * (dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (
                           M_SiO2 * (-dM_FeO_er - dM_FeSiO3_er) + M_m * (
                           dM_FeO_er + dM_FeSiO3_er))))) + dKFeSiO3_KFeSiO3 * (M_O * (M_Fe * M_FeO * (
        -4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Mg * (
                                                                                      M_Fe * (M_MgSiO3 * (M_FeO * (
                                                                                      4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                                          4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                              -4.0 * M_FeO - 4.0 * M_MgO)) + M_FeO * (
                                                                                      -4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                                      M_MgO * (
                                                                                      4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)))) + M_Si * (
                                                                               M_Fe * (M_FeO * (
                                                                               -M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                               M_MgO * (
                                                                               M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                       M_MgO * (M_FeO * (
                                                                                       M_SiO2 - 4.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                       M_FeO * (
                                                                                       9.0 * M_MgO + M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                       9.0 * M_SiO2 - 9.0 * M_m) - 9.0 * M_SiO2 * M_m))) + M_FeO * M_O * (
                                                                               -9.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                               M_MgO * (
                                                                               9.0 * M_SiO2 - 9.0 * M_m) - 9.0 * M_SiO2 * M_m)) + M_Mg * (
                                                                               M_Fe * (M_MgSiO3 * (
                                                                               M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                                               M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                                                       M_FeO * (
                                                                                       M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                       M_SiO2 - 4.0 * M_m) + M_MgSiO3 * (
                                                                                       9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                       -M_FeO - M_MgO)) + M_FeO * (
                                                                               -M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                               M_MgO * (
                                                                               M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                                               M_MgO * (
                                                                               M_SiO2 - 4.0 * M_m) + M_MgSiO3 * (
                                                                               9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m) - 9.0 * M_SiO2 * M_m)))) + M_c * (
                                                                               M_Fe * (M_FeO * (
                                                                               M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                               M_MgO * (
                                                                               -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                       M_MgO * M_SiO2 * (
                                                                                       -M_FeO + M_m) + M_MgSiO3 * (
                                                                                       M_FeO * (
                                                                                       -M_MgO - M_SiO2) + M_MgO * (
                                                                                       -M_SiO2 + M_m) + M_SiO2 * M_m))) + M_FeO * M_O * (
                                                                               M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                               M_MgO * (
                                                                               -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_Mg * (
                                                                               M_Fe * (M_MgSiO3 * (
                                                                               M_FeO * (-M_SiO2 + M_m) + M_MgO * (
                                                                               -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                                                       M_MgSiO3 * (
                                                                                       -M_FeO - M_MgO + M_m) + M_SiO2 * (
                                                                                       -M_FeO - M_MgO + M_m)) + M_SiO2 * M_m * (
                                                                                       M_FeO + M_MgO)) + M_FeO * (
                                                                               M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                               M_MgO * (
                                                                               -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                                               M_MgSiO3 * (-M_MgO + M_m) + M_SiO2 * (
                                                                               -M_MgO + M_m)))) + M_Si * (M_Fe * (
                                                                               M_MgO * (M_FeO * (
                                                                               -M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                               M_FeO * (
                                                                               -4.0 * M_MgO - M_SiO2 + M_m) + M_MgO * (
                                                                               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_O * (
                                                                               M_MgO * (
                                                                               -M_FeO - M_SiO2 + M_m) + M_MgSiO3 * (
                                                                               -M_FeO - M_SiO2 + M_m))) + M_FeO * (
                                                                                                          4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                                                          M_MgO * (
                                                                                                          -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                          M_MgO * (
                                                                                                          -M_SiO2 + M_m) + M_MgSiO3 * (
                                                                                                          -M_SiO2 + M_m))) + M_Mg * (
                                                                                                          M_Fe * (
                                                                                                          M_FeO * (
                                                                                                          -M_SiO2 + M_m) + M_MgO * (
                                                                                                          -M_SiO2 + M_m) + M_MgSiO3 * (
                                                                                                          -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_O * (
                                                                                                          -M_FeO - M_MgO - M_MgSiO3 - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          -M_SiO2 + M_m) + M_MgSiO3 * (
                                                                                                          -4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_O * (
                                                                                                          -M_MgO - M_MgSiO3 - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m)))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_FeO_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_FeO * (M_Fe * (M_O * (M_MgO * (M_FeSiO3 * (
        M_SiO2 * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeO_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeSiO3 * (
        M_SiO2 * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
        4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (-4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                      -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er))) + dKFeO_KFeO * (
                                M_Mg * M_O * (
                                M_MgO * (M_FeSiO3 * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Si * (M_Mg * (
                                M_MgO * (M_FeSiO3 * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
                                M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                M_FeSiO3 * (9.0 * M_MgO + 6.0 * M_SiO2 - 15.0 * M_m) + M_MgO * (
                                M_SiO2 - 4.0 * M_m) + M_MgSiO3 * (
                                9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m) - 9.0 * M_SiO2 * M_m)) + M_O * (M_MgO * (
                                M_FeSiO3 * (6.0 * M_SiO2 - 15.0 * M_m) - 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeSiO3 * (
                                6.0 * M_SiO2 - 15.0 * M_m) + M_MgO * (
                                                                                                           9.0 * M_SiO2 - 9.0 * M_m) - 9.0 * M_SiO2 * M_m))) + M_c * (
                                M_Mg * (M_MgO * (M_FeSiO3 * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
                                M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                        M_FeSiO3 * (-M_MgO + M_m) + M_MgSiO3 * (-M_MgO + M_m) + M_SiO2 * (
                                        -M_MgO + M_m))) + M_O * (M_MgO * M_m * (M_FeSiO3 + M_SiO2) + M_MgSiO3 * (
                                M_MgO * (-M_SiO2 + M_m) + M_m * (M_FeSiO3 + M_SiO2))) + M_Si * (M_Mg * (
                                M_FeSiO3 * (-4.0 * M_MgO - 2.0 * M_SiO2 + 6.0 * M_m) + M_MgO * (
                                -M_SiO2 + M_m) + M_MgSiO3 * (-4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_O * (
                                -M_MgO - M_MgSiO3 - M_SiO2 + M_m) + 4.0 * M_SiO2 * M_m) + M_MgO * (M_FeSiO3 * (
                                -2.0 * M_SiO2 + 6.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeSiO3 * (
                                -2.0 * M_SiO2 + 6.0 * M_m) + M_MgO * (
                                                                                               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                M_MgO * (
                                                                                                -M_SiO2 + M_m) + M_MgSiO3 * (
                                                                                                -M_SiO2 + M_m)))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                        M_O * (M_Fe * (4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                        M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Mg * (
                               M_MgSiO3 * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                               4.0 * M_Fe + 4.0 * M_MgO))) + M_Si * (M_Fe * (
                        M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                        M_MgO * (2.0 * M_SiO2 + 10.0 * M_m) + M_MgSiO3 * (
                        -9.0 * M_MgO + 2.0 * M_SiO2 + 10.0 * M_m))) + M_Mg * (M_Fe * (
                        M_O * (2.0 * M_SiO2 + 10.0 * M_m) + M_SiO2 * M_m) + M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_MgO * (
                        -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (M_MgO * (-M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                        -9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_O * (
                                                                     9.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_MgO * (
                                                                     -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_c * (
                        M_Fe * (
                        -M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_MgO * (M_O + M_SiO2 - M_m) - M_SiO2 * M_m)) + M_Mg * (
                        M_MgSiO3 * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                        M_MgSiO3 * (M_MgO - M_m) + M_SiO2 * (M_MgO - M_m)) + M_SiO2 * M_m * (-M_Fe - M_MgO)) + M_O * (
                        -M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_Si * (M_Fe * (
                        M_MgO * (-M_SiO2 - 3.0 * M_m) + M_MgSiO3 * (4.0 * M_MgO - M_SiO2 - 3.0 * M_m) + M_O * (
                        M_MgO + M_MgSiO3)) + M_Mg * (M_Fe * (M_O - M_SiO2 - 3.0 * M_m) + M_MgO * (
                        M_SiO2 - M_m) + M_MgSiO3 * (4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                     M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) - 4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                                                                              M_MgO * (
                                                                                                              4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                              M_MgO * (
                                                                                                              M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                              M_SiO2 - M_m))))) + M_Mg * (
                        M_O * (M_Fe * (M_FeSiO3 * (
                        M_SiO2 * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                        -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                       M_SiO2 * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                       -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                       -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgO * (
                               M_FeSiO3 * (M_SiO2 * (
                               -4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                           -4.0 * dM_FeO_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                               -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeSiO3 * (
                        M_SiO2 * (-4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                        -4.0 * dM_FeO_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                        4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                           -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                     -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er))) + dKMgO_KMgO * (
                        M_Fe * M_O * (
                        M_MgO * (M_FeSiO3 * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                        M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Si * (M_Fe * (
                        M_MgO * (M_FeSiO3 * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
                        M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                        M_MgO * (-9.0 * M_FeSiO3 - M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                        -9.0 * M_MgO + 2.0 * M_SiO2 + 10.0 * M_m))) + M_FeSiO3 * M_O * (M_MgO * (
                        -3.0 * M_SiO2 - 6.0 * M_m) + M_MgSiO3 * (6.0 * M_SiO2 - 15.0 * M_m))) + M_c * (M_Fe * (M_MgO * (
                        M_FeSiO3 * (M_SiO2 - M_m) + M_O * (M_FeSiO3 + M_MgSiO3 + M_SiO2) - M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                               M_MgO * (
                                                                                                               M_SiO2 - M_m) - M_SiO2 * M_m)) + M_FeSiO3 * M_O * (
                                                                                                       M_MgO * M_SiO2 + M_MgSiO3 * M_m) + M_Si * (
                                                                                                       M_Fe * (M_MgO * (
                                                                                                       4.0 * M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                               4.0 * M_MgO - M_SiO2 - 3.0 * M_m) + M_O * (
                                                                                                               M_MgO + M_MgSiO3)) + M_FeSiO3 * (
                                                                                                       M_MgO * (
                                                                                                       2.0 * M_SiO2 + 2.0 * M_m) + M_MgSiO3 * (
                                                                                                       -2.0 * M_SiO2 + 6.0 * M_m)))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                        M_O * (M_Fe * M_FeSiO3 * M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) + M_Mg * (
                        4.0 * M_Fe * M_SiO2 * M_m + M_FeSiO3 * M_MgO * (4.0 * M_SiO2 - 4.0 * M_m))) + M_Si * (M_Mg * (
                        M_Fe * (M_O * (2.0 * M_SiO2 + 10.0 * M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                        M_MgO * (M_SiO2 - M_m) + M_O * (9.0 * M_MgO + 6.0 * M_SiO2 - 15.0 * M_m))) + M_MgO * (M_Fe * (
                        M_FeSiO3 * (M_SiO2 - M_m) + M_O * (
                        9.0 * M_FeSiO3 + 3.0 * M_SiO2 + 6.0 * M_m)) + M_FeSiO3 * M_O * (
                                                                                                              9.0 * M_SiO2 - 9.0 * M_m))) + M_c * (
                        M_Mg * (
                        -M_Fe * M_SiO2 * M_m + M_FeSiO3 * (M_MgO * (-M_SiO2 + M_m) + M_O * (-M_MgO + M_m))) + M_MgO * (
                        M_Fe * (M_FeSiO3 * (-M_SiO2 + M_m) + M_O * (-M_FeSiO3 - M_SiO2)) + M_FeSiO3 * M_O * (
                        -M_SiO2 + M_m)) + M_Si * (M_Mg * (M_Fe * (M_O - M_SiO2 - 3.0 * M_m) + M_FeSiO3 * (
                        -4.0 * M_MgO - 2.0 * M_SiO2 + 6.0 * M_m)) + M_MgO * (
                                                  M_Fe * (-4.0 * M_FeSiO3 - 2.0 * M_SiO2 - 2.0 * M_m) + M_FeSiO3 * (
                                                  -4.0 * M_SiO2 + 4.0 * M_m))))) + M_Si * (M_Fe * (M_MgO * (M_FeSiO3 * (
        M_SiO2 * (-1.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
        -1.0 * dM_FeO_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                            -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                                   M_FeSiO3 * (
                                                                                                   M_SiO2 * (
                                                                                                   -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                                   -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                   M_SiO2 * (
                                                                                                   1.0 * dM_FeO_er + 1.0 * dM_FeSiO3_er) + M_m * (
                                                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_O * (
                                                                                                   M_MgO * (M_FeSiO3 * (
                                                                                                   -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 15.0 * dM_MgO_er - 24.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                            -2 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                            -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 6.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                   M_FeSiO3 * (
                                                                                                   -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_MgO * (
                                                                                                   -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                   -2 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 3.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                   -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er - 6.0 * dM_SiO2_er)))) + M_Mg * (
                                                                                           M_Fe * (M_FeSiO3 * (
                                                                                           M_SiO2 * (
                                                                                           -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                           -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                   M_SiO2 * (
                                                                                                   -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                                   -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_O * (
                                                                                                   M_FeSiO3 * (
                                                                                                   -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                                                   -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                   -2 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                   -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                           M_FeSiO3 * (M_SiO2 * (
                                                                                           -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                       -1.0 * dM_FeO_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                           -dM_FeO_er - dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                           M_FeSiO3 * (M_SiO2 * (
                                                                                           -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                                       -1.0 * dM_FeO_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgO * (
                                                                                           M_SiO2 * (
                                                                                           dM_FeO_er + dM_FeSiO3_er) + M_m * (
                                                                                           -dM_FeO_er - dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                           -dM_FeO_er - dM_FeSiO3_er)) + M_O * (
                                                                                           M_FeSiO3 * (M_MgO * (
                                                                                           3.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                       -9.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_m * (
                                                                                                       -9.0 * dM_FeO_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er)) + M_MgO * (
                                                                                           M_SiO2 * (
                                                                                           dM_FeO_er + dM_FeSiO3_er) + M_m * (
                                                                                           -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                           M_FeSiO3 * (
                                                                                           -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_MgO * (
                                                                                           9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                                           4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                           -25.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                           -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er))) + M_O * (
                                                                                           M_MgO * (M_FeSiO3 * (
                                                                                           M_SiO2 * (
                                                                                           -9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 18.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_m * (
                                                                                           -9.0 * dM_FeO_er + 9.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                    -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                           M_FeSiO3 * (M_SiO2 * (
                                                                                           -9.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_m * (
                                                                                                       -9.0 * dM_FeO_er - 9.0 * dM_MgO_er + 9.0 * dM_SiO2_er)) + M_MgO * (
                                                                                           M_SiO2 * (
                                                                                           9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er) + M_m * (
                                                                                           -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                           -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er))) + dKSiO2_KSiO2 * (
                                                                                           M_O * (M_Fe * (M_MgO * (
                                                                                           M_FeSiO3 * (
                                                                                           -4.0 * M_SiO2 + 10.0 * M_m) + 6.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                          M_FeSiO3 * (
                                                                                                          -4.0 * M_SiO2 + 10.0 * M_m) + M_MgO * (
                                                                                                          -6.0 * M_SiO2 + 6.0 * M_m) + 6.0 * M_SiO2 * M_m)) + M_Mg * (
                                                                                                  M_Fe * (M_FeSiO3 * (
                                                                                                  -4.0 * M_SiO2 + 10.0 * M_m) + M_MgSiO3 * (
                                                                                                          -4.0 * M_SiO2 + 10.0 * M_m) + 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                  M_MgO * (
                                                                                                  2.0 * M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                                                                                                  -4.0 * M_SiO2 + 10.0 * M_m)))) + M_c * (
                                                                                           M_Fe * (M_MgO * (M_FeSiO3 * (
                                                                                           M_SiO2 - 3.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                   M_FeSiO3 * (
                                                                                                   M_SiO2 - 3.0 * M_m) + M_MgO * (
                                                                                                   2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeSiO3 + M_SiO2) + M_MgSiO3 * (
                                                                                                   M_FeSiO3 + M_SiO2))) + M_FeSiO3 * M_O * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_SiO2 - M_m)) + M_Mg * (
                                                                                           M_Fe * (M_FeSiO3 * (
                                                                                           M_SiO2 - 3.0 * M_m) + M_MgSiO3 * (
                                                                                                   M_SiO2 - 3.0 * M_m) + M_O * (
                                                                                                   M_FeSiO3 + M_MgSiO3 + M_SiO2) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_MgO * (
                                                                                           -M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_SiO2 - 3.0 * M_m) + M_O * (
                                                                                           M_MgO + M_MgSiO3 + M_SiO2 - M_m)))))) + M_c * (
                        M_Fe * (M_MgO * (M_FeSiO3 * (
                        M_SiO2 * (dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                         dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeSiO3 * (
                        M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                        -dM_FeO_er - 1.0 * dM_FeSiO3_er) + M_m * (dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                        dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_O * (
                                M_MgO * (
                                M_FeSiO3 * (dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_SiO2 * (
                                dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                M_FeSiO3 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_MgO * (
                                dM_MgO_er + dM_MgSiO3_er) + M_SiO2 * (
                                dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)))) + M_Mg * (M_Fe * (M_FeSiO3 * (
                        M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgSiO3 * (M_SiO2 * (
                        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                 dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_O * (
                                                                                               M_FeSiO3 * (
                                                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_MgSiO3 * (
                                                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_SiO2 * (
                                                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                               dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                       M_FeSiO3 * (M_SiO2 * (
                                                                                       1.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                                                                   dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                       M_FeSiO3 * (M_SiO2 * (
                                                                                       dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                   dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgO * (
                                                                                       M_SiO2 * (
                                                                                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_O * (
                                                                                       M_FeSiO3 * (M_MgO * (
                                                                                       -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er) + M_SiO2 * (
                                                                                                   dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                   dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er)) + M_MgSiO3 * (
                                                                                       M_FeSiO3 * (
                                                                                       dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_MgO * (
                                                                                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * (
                                                                                       M_MgO * (
                                                                                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                       dM_FeO_er + dM_FeSiO3_er)))) + M_O * (
                        M_MgO * (M_FeSiO3 * (
                        M_SiO2 * (dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er)) + M_SiO2 * M_m * (
                                 dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (M_FeSiO3 * (
                        M_SiO2 * (dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                        dM_FeO_er + dM_MgO_er - dM_SiO2_er)) + M_MgO * (M_SiO2 * (-dM_FeO_er - dM_FeSiO3_er) + M_m * (
                        dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (dM_FeO_er + dM_FeSiO3_er))) + M_Si * (M_Fe * (
                        M_MgO * (M_FeSiO3 * (
                        2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 10.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_SiO2 * (
                                 dM_FeO_er + 3.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 4.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                 dM_FeO_er + 3.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_MgSiO3 * (
                        M_FeSiO3 * (
                        2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_MgO * (
                        4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                        dM_FeO_er + 3.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                        dM_FeO_er + 3.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er + 2.0 * dM_SiO2_er)) + M_O * (
                        M_MgO * (dM_MgO_er + dM_MgSiO3_er) + M_MgSiO3 * (dM_MgO_er + dM_MgSiO3_er))) + M_Mg * (M_Fe * (
                        M_FeSiO3 * (
                        2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_MgSiO3 * (
                        2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_SiO2 * (
                        dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                        dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                               M_MgO * (
                                                                                                               -2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                               4.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                                               4.0 * dM_FeO_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                               M_SiO2 * (
                                                                                                               -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                               dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                                               M_FeSiO3 * (
                                                                                                               2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_MgO * (
                                                                                                               -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                                                               -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                               9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er)) + M_O * (
                                                                                                               M_MgO * (
                                                                                                               -dM_FeO_er - dM_FeSiO3_er) + M_MgSiO3 * (
                                                                                                               -dM_FeO_er - dM_FeSiO3_er) + M_SiO2 * (
                                                                                                               -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                               dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                               4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_MgO * (
                                                                                                           M_FeSiO3 * (
                                                                                                           M_SiO2 * (
                                                                                                           4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                                           4.0 * dM_FeO_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                           4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                                           M_FeSiO3 * (
                                                                                                           M_SiO2 * (
                                                                                                           4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                                           4.0 * dM_FeO_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                           M_SiO2 * (
                                                                                                           -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                                           4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                           4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           M_SiO2 * (
                                                                                                           -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                           dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                                           M_SiO2 * (
                                                                                                           -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                           dM_FeO_er + dM_FeSiO3_er)))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_Si_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_Si * (M_Fe * (M_O * (M_MgO * (M_FeO * (
        M_SiO2 * (-2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                    4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                    10.0 * dM_FeO_er - 10.0 * dM_MgSiO3_er - 10.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                               6.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
        -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                 -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                 4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                 10.0 * dM_FeO_er + 10.0 * dM_MgO_er - 10.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                    M_SiO2 * (
                                                                                                    -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er) + M_m * (
                                                                                                    6.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                    6.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er))) + dKFeO_KFeO * (
                               M_O * (M_Mg * (
                               M_FeO * (M_MgSiO3 * (4.0 * M_SiO2 - 10.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                               M_FeO * (6.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                               2.0 * M_SiO2 + 4.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgO * (
                                      -6.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                      M_FeO * (6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                      M_FeO * (M_MgO * (6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                      M_FeO * (6.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                                      6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m))) + M_c * (M_Mg * (
                               M_FeO * (M_MgSiO3 * (-M_SiO2 + 3.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                               M_FeO * (-2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                               -M_SiO2 - M_m) + 2.0 * M_SiO2 * M_m) + M_O * (
                               M_FeO * (-M_MgSiO3 - M_SiO2) + M_FeSiO3 * (M_MgO - M_m))) + M_MgO * (
                                                                                                 2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (M_MgO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_O * (
                                                                                                 M_MgO * (
                                                                                                 -M_FeO * M_SiO2 - M_FeSiO3 * M_m) + M_MgSiO3 * (
                                                                                                 -M_FeO * M_SiO2 - M_FeSiO3 * M_m))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                       M_O * (M_Fe * (M_MgO * (M_FeO * (2.0 * M_SiO2 + 4.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                       M_FeO * (2.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                       6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_Mg * (M_Fe * (
                       M_FeO * (2.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                       2.0 * M_SiO2 + 4.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeO * (M_MgO * (
                       2.0 * M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (-4.0 * M_SiO2 + 10.0 * M_m)))) + M_c * (M_Fe * (
                       M_MgO * (M_FeO * (-M_SiO2 - M_m) + 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                       M_FeO * (-M_SiO2 - M_m) + M_MgO * (-2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_O * (
                       M_MgO * (M_FeO - M_m) + M_MgSiO3 * (M_FeO - M_m))) + M_FeO * M_O * (M_MgO * (
                       M_SiO2 - M_m) + M_MgSiO3 * (M_SiO2 - M_m)) + M_Mg * (M_Fe * (
                       M_FeO * (-M_SiO2 - M_m) + M_MgO * (-M_SiO2 - M_m) + M_O * (
                       M_FeO + M_MgO - M_m) + 2.0 * M_SiO2 * M_m) + M_FeO * (M_MgO * (-M_SiO2 - M_m) + M_MgSiO3 * (
                       M_SiO2 - 3.0 * M_m) + M_O * (M_MgO + M_MgSiO3 + M_SiO2 - M_m))))) + M_Mg * (M_O * (M_Fe * (
        M_FeO * (
        M_SiO2 * (-2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
        M_FeO * (-6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_MgO * (
        -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
        4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
        10.0 * dM_FeO_er + 10.0 * dM_MgO_er - 10.0 * dM_SiO2_er)) + M_MgO * (
        M_SiO2 * (-2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_MgSiO3 * (
        M_FeO * (-6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_MgO * (
        -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
        4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
        10.0 * dM_FeO_er + 10.0 * dM_MgO_er - 10.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
        6.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er)) + M_FeO * (M_MgO * (
        M_SiO2 * (-2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
        -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                 6.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          -6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_m * (
                                                                                                          6.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                                          M_SiO2 * (
                                                                                                          -2.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                                          4.0 * dM_FeO_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                          6.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                                          -10.0 * dM_FeSiO3_er + 10.0 * dM_MgO_er - 10.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                          -6.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                                                                          10.0 * dM_FeO_er + 10.0 * dM_MgO_er - 10.0 * dM_SiO2_er)))) + dKMgO_KMgO * (
                                                                                                   M_O * (M_Fe * (
                                                                                                   M_MgO * (M_FeSiO3 * (
                                                                                                   4.0 * M_SiO2 - 10.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                   M_FeO * (
                                                                                                   2.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                                   6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgO * (
                                                                                                          -6.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          6.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                                                                                                          6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m))) + M_c * (
                                                                                                   M_Fe * (M_MgO * (
                                                                                                   M_FeSiO3 * (
                                                                                                   -M_SiO2 + 3.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -M_SiO2 - M_m) + M_MgO * (
                                                                                                           -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           -M_FeSiO3 - M_SiO2) + M_MgSiO3 * (
                                                                                                           M_FeO - M_m))) + M_MgO * (
                                                                                                   2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                   M_FeO * (
                                                                                                   -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                   M_FeO * (M_MgO * (
                                                                                                   -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                   M_FeO * (
                                                                                                   -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                                   -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_O * (
                                                                                                   M_MgO * M_SiO2 * (
                                                                                                   -M_FeO - M_FeSiO3) + M_MgSiO3 * M_m * (
                                                                                                   -M_FeO - M_FeSiO3))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                       M_O * (M_Fe * M_MgO * (
                       M_FeO * (2.0 * M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (-4.0 * M_SiO2 + 10.0 * M_m)) + M_Mg * (M_Fe * (
                       M_FeO * (2.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                       2.0 * M_SiO2 + 4.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeO * (M_MgO * (
                       2.0 * M_SiO2 + 4.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                       6.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (2.0 * M_SiO2 + 4.0 * M_m) - 6.0 * M_SiO2 * M_m))) + M_c * (
                       M_Mg * (M_Fe * (M_FeO * (-M_SiO2 - M_m) + M_MgO * (-M_SiO2 - M_m) + M_O * (
                       M_FeO + M_MgO - M_m) + 2.0 * M_SiO2 * M_m) + M_FeO * (
                               M_MgO * (-M_SiO2 - M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                               M_FeO * (-2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                               -M_SiO2 - M_m) + 2.0 * M_SiO2 * M_m) + M_O * (
                               M_FeO * (M_MgO - M_m) + M_FeSiO3 * (M_MgO - M_m))) + M_MgO * (M_Fe * (
                       M_FeO * (-M_SiO2 - M_m) + M_FeSiO3 * (M_SiO2 - 3.0 * M_m) + M_O * (
                       M_FeO + M_FeSiO3 + M_SiO2 - M_m)) + M_O * (M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
                       M_SiO2 - M_m))))) + M_c * (M_Fe * (M_MgO * (M_FeO * (
        M_SiO2 * (dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                 -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                 -3.0 * dM_FeO_er + 3.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                   -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                          M_FeO * (M_MgO * (
                                                          2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                   dM_FeO_er + 2.0 * dM_FeSiO3_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                                   dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                          M_FeO * (
                                                          2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                          2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                          -3.0 * dM_FeO_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er)) + M_MgO * (
                                                          M_SiO2 * (2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er) + M_m * (
                                                          -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                          -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er)) + M_O * (M_MgO * (
        M_FeO * (-dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_FeSiO3 * (
        -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_SiO2 * (
        -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
        dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (M_FeO * (
        -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er) + M_FeSiO3 * (-dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_SiO2 * (
                                                                 -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                 dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)))) + M_Mg * (
                                                  M_Fe * (M_FeO * (M_SiO2 * (
                                                  dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                                   dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                          M_FeO * (
                                                          2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                          2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                          -3.0 * dM_FeO_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er)) + M_MgO * (
                                                          M_SiO2 * (
                                                          dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                                          M_FeO * (
                                                          2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                          2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                          -3.0 * dM_FeO_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er)) + M_O * (
                                                          M_FeO * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_FeSiO3 * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgSiO3 * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_SiO2 * (
                                                          -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                          -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er)) + M_FeO * (
                                                  M_MgO * (M_SiO2 * (
                                                  dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                           dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                  -2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                                                  M_MgO * (
                                                  2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                  2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_m * (
                                                  -2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                                                  1.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                                                                                     -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                        -2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                                                  M_FeO * (M_MgO * (
                                                  2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                           -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                           3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                  M_FeO * (
                                                  2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                  2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                  -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                  -3.0 * dM_FeO_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er))) + M_O * (
                                                  M_FeO * (
                                                  M_MgO * (-dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_SiO2 * (
                                                  -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                  dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                  M_MgO * (dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er) + M_SiO2 * (
                                                  -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                  -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (M_FeO * (
                                                  -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_FeSiO3 * (
                                                                                                         -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er)))) + M_O * (
                                                  M_MgO * (M_FeO * (M_SiO2 * (
                                                  -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                    dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                           M_SiO2 * (
                                                           -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                           -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er))) + M_MgSiO3 * (
                                                  M_FeO * (M_SiO2 * (
                                                  -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                           dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                  M_SiO2 * (-dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                  -dM_FeO_er - dM_MgO_er + dM_SiO2_er))))) + dKSiO2_KSiO2 * (M_O * (
        M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                     M_FeO * (
                                                                                     -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                     -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                     4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                             4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                             M_FeO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                             M_FeO * (M_MgO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                             M_FeO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                             -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_c * (
                                                                                                             M_Fe * (
                                                                                                             M_MgO * (
                                                                                                             -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) + M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                                             M_MgO * (
                                                                                                             M_FeSiO3 * (
                                                                                                             M_FeO - M_m) + M_SiO2 * (
                                                                                                             M_FeO - M_m)) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                                             M_FeO + M_MgO - M_m) + M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (
                                                                                                             M_Fe * (
                                                                                                             M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) + M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) + M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                                                                             M_FeSiO3 * (
                                                                                                             M_FeO + M_MgO - M_m) + M_MgSiO3 * (
                                                                                                             M_FeO + M_MgO - M_m) + M_SiO2 * (
                                                                                                             M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (
                                                                                                             -M_FeO - M_MgO)) + M_MgO * (
                                                                                                             -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) + M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                                             M_FeO * M_SiO2 * (
                                                                                                             M_MgO - M_m) + M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                                                                             M_MgO - M_m)) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_MgO - M_m) + M_FeSiO3 * (
                                                                                                             M_FeO + M_MgO - M_m)))) + M_O * (
                                                                                                             M_MgO * (
                                                                                                             -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                             M_FeO * (
                                                                                                             M_SiO2 - M_m) + M_MgO * (
                                                                                                             M_SiO2 - M_m) - M_SiO2 * M_m)))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_Fe_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_Fe * (M_FeSiO3 * dKFeSiO3_KFeSiO3 * (M_Mg * M_O * (4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Si * (
                                                      M_Mg * (M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                      M_FeO * (-M_SiO2 + M_m) + M_MgO * (
                                                      -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                              M_FeO * (-3.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                                                              -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                                                              -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_O * (
                                                      M_MgO * (M_FeO * (
                                                      -3.0 * M_SiO2 - 6.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                      M_FeO * (-3.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                                                      -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_c * (
                                                      M_Mg * (-M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                                                      M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                      M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                              M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
                                                              M_FeO + M_MgO - M_m))) + M_O * (
                                                      M_MgO * M_SiO2 * (M_FeO - M_m) + M_MgSiO3 * (
                                                      M_MgO * (M_SiO2 - M_m) + M_SiO2 * (M_FeO - M_m))) + M_Si * (
                                                      M_Mg * (M_FeO * (2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                      M_SiO2 - M_m) + M_MgSiO3 * (
                                                              4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                              M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgO * (
                                                      M_FeO * (
                                                      2.0 * M_SiO2 + 2.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                      M_FeO * (2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                      4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                      M_MgO * (M_SiO2 - M_m) + M_MgSiO3 * (M_SiO2 - M_m))))) + M_Mg * (
                       M_O * (M_FeSiO3 * (M_FeO * (M_SiO2 * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                       4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                       -4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
                                                                         -4.0 * dM_FeO_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                          4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
                       M_SiO2 * (4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                       -4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er - 4.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                       4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                             -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                               -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                              M_FeO * (4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_MgO * (
                              -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er))) + dKMgO_KMgO * (M_O * (M_MgO * (
                       -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
                       M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                       4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m))) + M_Si * (
                                                                                       M_MgO * (
                                                                                       -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                       M_FeO * (
                                                                                       M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                       M_FeO * (M_MgO * (
                                                                                       M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                       M_FeO * (
                                                                                       M_SiO2 - M_m) + M_MgO * (
                                                                                       M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                       M_MgO * (M_FeO * (
                                                                                       M_SiO2 - 4.0 * M_m) + M_FeSiO3 * (
                                                                                                9.0 * M_FeO - 2.0 * M_SiO2 - 10.0 * M_m)) + M_MgSiO3 * (
                                                                                       M_FeO * (
                                                                                       9.0 * M_MgO - 2.0 * M_SiO2 - 10.0 * M_m) + M_FeSiO3 * (
                                                                                       9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m)))) + M_c * (
                                                                                       M_MgO * (
                                                                                       M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                       M_FeO * (
                                                                                       -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                       M_FeO * (M_MgO * (
                                                                                       -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                       M_FeO * (
                                                                                       -M_SiO2 + M_m) + M_MgO * (
                                                                                       -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                       M_FeO * M_MgO * (
                                                                                       -M_FeSiO3 - M_SiO2) + M_MgSiO3 * (
                                                                                       -M_FeO * M_MgO + M_FeSiO3 * (
                                                                                       -M_FeO - M_MgO + M_m))) + M_Si * (
                                                                                       M_MgO * (M_FeO * (
                                                                                       -M_SiO2 + M_m) + M_FeSiO3 * (
                                                                                                -4.0 * M_FeO + M_SiO2 + 3.0 * M_m)) + M_MgSiO3 * (
                                                                                       M_FeO * (
                                                                                       -4.0 * M_MgO + M_SiO2 + 3.0 * M_m) + M_FeSiO3 * (
                                                                                       -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m)) + M_O * (
                                                                                       M_MgO * (
                                                                                       -M_FeO - M_FeSiO3) + M_MgSiO3 * (
                                                                                       -M_FeO - M_FeSiO3)))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                       M_Mg * M_O * (-4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                       4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Si * (M_Mg * (
                       -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                       M_FeO * (-2.0 * M_SiO2 - 10.0 * M_m) + M_FeSiO3 * (
                       9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m))) + M_MgO * M_O * (M_FeO * (
                       -3.0 * M_SiO2 - 6.0 * M_m) + M_FeSiO3 * (6.0 * M_SiO2 - 15.0 * M_m))) + M_c * (M_Mg * (
                       M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_O * (
                       -M_FeO - M_MgO + M_m) + M_SiO2 * M_m)) + M_MgO * M_O * (
                                                                                                      M_FeO * M_SiO2 + M_FeSiO3 * M_m) + M_Si * (
                                                                                                      M_Mg * (M_FeO * (
                                                                                                      M_SiO2 + 3.0 * M_m) + M_FeSiO3 * (
                                                                                                              -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_O * (
                                                                                                              -M_FeO - M_FeSiO3)) + M_MgO * (
                                                                                                      M_FeO * (
                                                                                                      2.0 * M_SiO2 + 2.0 * M_m) + M_FeSiO3 * (
                                                                                                      -2.0 * M_SiO2 + 6.0 * M_m))))) + M_Si * (
                       M_Mg * (M_FeSiO3 * (M_FeO * (M_SiO2 * (-1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                       1.0 * dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                       -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                         -1.0 * dM_FeO_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                           1.0 * dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
                       M_SiO2 * (dM_FeO_er + 2.0 * dM_FeSiO3_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                       -1.0 * dM_FeSiO3_er + 1.0 * dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                       dM_FeO_er + dM_FeSiO3_er) + M_m * (-dM_FeO_er - dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                -dM_FeO_er - dM_FeSiO3_er)) + M_O * (
                               M_FeO * (M_SiO2 * (
                               3.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                                        6.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 10.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_FeSiO3 * (
                               M_FeO * (9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er) + M_MgO * (
                               3.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_SiO2 * (
                               -6.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
                               -15.0 * dM_FeO_er + 10.0 * dM_MgO_er + 25.0 * dM_MgSiO3_er + 15.0 * dM_SiO2_er)) + M_MgO * (
                               M_SiO2 * (dM_FeO_er + dM_FeSiO3_er) + M_m * (
                               -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (
                               15.0 * dM_FeO_er + 24.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_MgO * (
                                                                                     9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                                     4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                     -25.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                               -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                               M_FeO * (1.0 * dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                               -dM_FeO_er - dM_FeSiO3_er))) + M_O * (M_MgO * (M_FeO * (M_SiO2 * (
                       3.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                                                                                       6.0 * dM_FeSiO3_er + 6.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                              M_FeO * (
                                                                              9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                              -6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 12.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
                                                                              -15.0 * dM_FeO_er + 15.0 * dM_MgSiO3_er + 15.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                              -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                     M_FeO * (M_MgO * (
                                                                     9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                              3.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 3.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                                                                              6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er + 6.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                     M_FeO * (
                                                                     9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_MgO * (
                                                                     9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                     -6.0 * dM_FeSiO3_er - 6.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
                                                                     -15.0 * dM_FeO_er - 15.0 * dM_MgO_er + 15.0 * dM_SiO2_er)) + M_MgO * (
                                                                     M_SiO2 * (
                                                                     9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er) + M_m * (
                                                                     -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                     -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er))) + dKSiO2_KSiO2 * (
                       M_O * (M_Mg * (
                       M_FeO * (M_MgSiO3 * (4.0 * M_SiO2 - 10.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                       M_FeO * (6.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                       2.0 * M_SiO2 + 4.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgO * (
                              -6.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                              M_FeO * (6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (6.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                              6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m))) + M_c * (M_Mg * (
                       M_FeO * (M_MgSiO3 * (-M_SiO2 + 3.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                       M_FeO * (-2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (-M_SiO2 - M_m) + 2.0 * M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_MgSiO3 - M_SiO2) + M_FeSiO3 * (M_MgO - M_m))) + M_MgO * (
                                                                                         2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (M_MgO * (
                                                                                         -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                         M_FeO * (
                                                                                         -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                         -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_O * (
                                                                                         M_MgO * (
                                                                                         -M_FeO * M_SiO2 - M_FeSiO3 * M_m) + M_MgSiO3 * (
                                                                                         -M_FeO * M_SiO2 - M_FeSiO3 * M_m))))) + M_c * (
                       M_Mg * (M_FeSiO3 * (M_FeO * (
                       M_SiO2 * (dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_m * (-dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_MgO * (
                                           M_SiO2 * (
                                           1.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                                           dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er)) + M_SiO2 * M_m * (
                                           -dM_MgO_er - 1.0 * dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
                       M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                       1.0 * dM_FeSiO3_er - dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                           dM_FeO_er + dM_FeSiO3_er)) + M_O * (
                               M_FeSiO3 * (M_FeO * (-dM_FeO_er - dM_FeSiO3_er) + M_MgO * (
                               -dM_FeO_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                           dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er)) + M_MgSiO3 * (
                               M_FeO * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                               -dM_FeO_er - dM_FeSiO3_er) + M_m * (dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * (
                               M_FeO * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                               -dM_FeO_er - dM_FeSiO3_er) + M_m * (dM_FeO_er + dM_FeSiO3_er))) + M_SiO2 * M_m * (
                               M_FeO * (-dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                               dM_FeO_er + dM_FeSiO3_er))) + M_O * (M_MgO * (M_FeSiO3 * (
                       M_FeO * (-dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_m * (
                       dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er)) + M_SiO2 * (M_FeO * (
                       -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                           dM_FeO_er + dM_FeSiO3_er))) + M_MgSiO3 * (
                                                                    M_FeO * (M_MgO * (
                                                                    -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                                                                             -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er)) + M_FeSiO3 * (
                                                                    M_FeO * (
                                                                    -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_MgO * (
                                                                    -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                    dM_FeO_er + dM_MgO_er - dM_SiO2_er)) + M_MgO * (
                                                                    M_SiO2 * (-dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                    dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                    dM_FeO_er + dM_FeSiO3_er))) + M_Si * (M_Mg * (
                       M_FeO * (M_SiO2 * (
                       -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                -2.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                       M_FeO * (-4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_MgO * (
                       -2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_SiO2 * (
                       2.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                       6.0 * dM_FeO_er - 3.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)) + M_MgO * (
                       M_SiO2 * (-dM_FeO_er - dM_FeSiO3_er) + M_m * (dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (
                       -6.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_MgO * (
                                                                                                              -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_SiO2 * (
                                                                                                              -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                              9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er)) + M_O * (
                       M_FeO * (-dM_FeO_er - dM_FeSiO3_er) + M_FeSiO3 * (-dM_FeO_er - dM_FeSiO3_er) + M_MgO * (
                       -dM_FeO_er - dM_FeSiO3_er) + M_MgSiO3 * (-dM_FeO_er - dM_FeSiO3_er) + M_SiO2 * (
                       -dM_FeO_er - dM_FeSiO3_er) + M_m * (dM_FeO_er + dM_FeSiO3_er)) + M_SiO2 * M_m * (
                       4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_MgO * (M_FeO * (M_SiO2 * (
                       -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                  -2.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                         M_FeO * (
                                                                         -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                         2.0 * dM_FeSiO3_er + 2 * dM_MgO_er + 4.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                         6.0 * dM_FeO_er - 6.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                         4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                                          -2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                          -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          2.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                                          6.0 * dM_FeO_er + 6.0 * dM_MgO_er - 6.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                          M_SiO2 * (
                                                                                                          -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_m * (
                                                                                                          4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                          4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_O * (
                                                                                                          M_MgO * (
                                                                                                          M_FeO * (
                                                                                                          -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_FeSiO3 * (
                                                                                                          -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                          dM_FeO_er + dM_FeSiO3_er)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_FeSiO3 * (
                                                                                                          -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          -dM_FeO_er - dM_FeSiO3_er) + M_m * (
                                                                                                          dM_FeO_er + dM_FeSiO3_er))))) + dKFeO_KFeO * (
                       M_Mg * M_O * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                     M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                     M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                     -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Si * (M_Mg * (M_MgO * (
                       M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          -M_SiO2 + M_m) + M_MgO * (
                                                                                                          -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                                          M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          -9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                                          -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          -9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
                                                                                                          -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (
                                                                                                  M_MgO * (
                                                                                                  9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                  M_FeO * (M_MgO * (
                                                                                                  -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                                  -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
                       M_Mg * (M_MgO * (
                       -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                               M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                               M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (M_MgO - M_m)) + M_MgSiO3 * (
                               M_FeO * (M_MgO - M_m) + M_FeSiO3 * (M_FeO + M_MgO - M_m)))) + M_O * (M_MgO * (
                       -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                    M_FeO * (M_MgO * (
                                                                                                    M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                    M_FeO * (
                                                                                                    M_SiO2 - M_m) + M_MgO * (
                                                                                                    M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (
                       M_Mg * (M_FeO * (M_MgO * (M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                       M_FeO * (4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                       M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                               M_FeO * (4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                               4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                               M_FeO * (M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                               M_FeO + M_FeSiO3))) + M_MgO * (-4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                       M_FeO * (M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                       M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                       4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                       M_MgO * (M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (M_SiO2 - M_m)) + M_MgSiO3 * (
                       M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (M_SiO2 - M_m))))))) / (M_O * (M_Fe * (M_MgO * (
        4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
        -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (
        M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
        -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
                                                                                   M_Fe * (M_MgO * (
                                                                                   M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -M_SiO2 + M_m) + M_MgO * (
                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           -M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                                    -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
                                                                                   M_Fe * (M_FeSiO3 * (
                                                                                   M_FeO * (-M_SiO2 + M_m) + M_MgO * (
                                                                                   -M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           -M_SiO2 + M_m) + M_MgO * (
                                                                                           -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                                                           M_FeO * (
                                                                                           -M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                           -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                                                                                           -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                           M_FeO + M_MgO)) + M_MgO * (
                                                                                   M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                   M_FeO * (M_MgO * (
                                                                                   -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                   M_FeO * (-M_SiO2 + M_m) + M_MgO * (
                                                                                   -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                   M_FeO * (M_MgO * (
                                                                                   -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   -9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                   -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                   M_FeO * (
                                                                                   -9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
                                                                                   -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (
                                                                                   M_MgO * (
                                                                                   9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                   M_FeO * (M_MgO * (
                                                                                   -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                   -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
                                                                                   M_Fe * (M_MgO * (
                                                                                   -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeSiO3 * (
                                                                                           M_FeO - M_m) + M_SiO2 * (
                                                                                                    M_FeO - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                           M_FeO + M_MgO - M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (
                                                                                   M_Fe * (M_FeSiO3 * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                                                                                           M_FeSiO3 * (
                                                                                           M_FeO + M_MgO - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_MgO - M_m) + M_SiO2 * (
                                                                                           M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (
                                                                                           -M_FeO - M_MgO)) + M_MgO * (
                                                                                   -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                   M_FeO * (M_MgO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                   M_FeO * M_SiO2 * (
                                                                                   M_MgO - M_m) + M_FeSiO3 * (M_FeO * (
                                                                                   M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                                                                              M_MgO - M_m)) + M_MgSiO3 * (
                                                                                   M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                                                   M_FeO + M_MgO - M_m)))) + M_O * (
                                                                                   M_MgO * (
                                                                                   -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                   M_FeO * (M_MgO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                                                   M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (
                                                                                   M_Fe * (M_MgO * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                   4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                           M_MgO * (
                                                                                           M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                   M_Fe * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                   M_MgO * (
                                                                                   M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                   M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                   M_FeO * (
                                                                                   4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                   M_FeO * (
                                                                                   M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                   M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                   M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                   -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                   M_FeO * (M_MgO * (
                                                                                   4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                   M_FeO * (
                                                                                   4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                   4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                   M_MgO * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                   M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                   M_SiO2 - M_m))))))

    def dM_Mg_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_Mg * (M_Fe * (M_O * (M_FeSiO3 * (M_FeO * (
        M_SiO2 * (4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgO * (
                                                  M_SiO2 * (
                                                  4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_m * (
                                                  4.0 * dM_FeO_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                  -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
        M_SiO2 * (-4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_m * (
        4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er + 4.0 * dM_SiO2_er)) + M_MgO * (M_SiO2 * (
        -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er) + M_m * (4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                                        4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                      M_FeO * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                      4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er))) + dKFeO_KFeO * (M_O * (M_MgO * (
        -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
        M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
        M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
        4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m))) + M_Si * (M_MgO * (
        -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
        M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (
        M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (M_MgO * (
        M_FeO * (M_SiO2 - 4.0 * M_m) + M_FeSiO3 * (9.0 * M_FeO - 2.0 * M_SiO2 - 10.0 * M_m)) + M_MgSiO3 * (
                                                M_FeO * (9.0 * M_MgO - 2.0 * M_SiO2 - 10.0 * M_m) + M_FeSiO3 * (
                                                9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m)))) + M_c * (
                                                                                              M_MgO * (
                                                                                              M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                              M_FeO * (
                                                                                              -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                              M_FeO * (M_MgO * (
                                                                                              -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                              M_FeO * (
                                                                                              -M_SiO2 + M_m) + M_MgO * (
                                                                                              -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                              M_FeO * M_MgO * (
                                                                                              -M_FeSiO3 - M_SiO2) + M_MgSiO3 * (
                                                                                              -M_FeO * M_MgO + M_FeSiO3 * (
                                                                                              -M_FeO - M_MgO + M_m))) + M_Si * (
                                                                                              M_MgO * (M_FeO * (
                                                                                              -M_SiO2 + M_m) + M_FeSiO3 * (
                                                                                                       -4.0 * M_FeO + M_SiO2 + 3.0 * M_m)) + M_MgSiO3 * (
                                                                                              M_FeO * (
                                                                                              -4.0 * M_MgO + M_SiO2 + 3.0 * M_m) + M_FeSiO3 * (
                                                                                              -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m)) + M_O * (
                                                                                              M_MgO * (
                                                                                              -M_FeO - M_FeSiO3) + M_MgSiO3 * (
                                                                                              -M_FeO - M_FeSiO3)))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                       M_Fe * M_O * (-4.0 * M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                       M_FeO * (4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                       4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_Si * (M_Fe * (
                       -M_MgO * M_SiO2 * M_m + M_MgSiO3 * (
                       M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                       M_MgO * (-2.0 * M_SiO2 - 10.0 * M_m) + M_MgSiO3 * (
                       9.0 * M_FeO + 9.0 * M_MgO + 4.0 * M_SiO2 - 25.0 * M_m))) + M_FeO * M_O * (M_MgO * (
                       -3.0 * M_SiO2 - 6.0 * M_m) + M_MgSiO3 * (6.0 * M_SiO2 - 15.0 * M_m))) + M_c * (M_Fe * (
                       M_MgO * M_SiO2 * M_m + M_MgSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_O * (
                       -M_FeO - M_MgO + M_m) + M_SiO2 * M_m)) + M_FeO * M_O * (
                                                                                                      M_MgO * M_SiO2 + M_MgSiO3 * M_m) + M_Si * (
                                                                                                      M_Fe * (M_MgO * (
                                                                                                      M_SiO2 + 3.0 * M_m) + M_MgSiO3 * (
                                                                                                              -4.0 * M_FeO - 4.0 * M_MgO - M_SiO2 + 9.0 * M_m) + M_O * (
                                                                                                              -M_MgO - M_MgSiO3)) + M_FeO * (
                                                                                                      M_MgO * (
                                                                                                      2.0 * M_SiO2 + 2.0 * M_m) + M_MgSiO3 * (
                                                                                                      -2.0 * M_SiO2 + 6.0 * M_m))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                       M_Fe * M_O * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                       -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_Si * (M_Fe * (
                       M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -3.0 * M_SiO2 - 6.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_O * (M_FeO * (
                       M_MgO * (-3.0 * M_SiO2 - 6.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                       -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                               -3.0 * M_SiO2 - 6.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_c * (
                       M_Fe * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                       M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
                               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (M_FeO + M_MgO - M_m))) + M_O * (
                       M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                       M_FeO * (M_SiO2 - M_m) + M_SiO2 * (M_MgO - M_m))) + M_Si * (M_Fe * (
                       M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                       2.0 * M_SiO2 + 2.0 * M_m) + M_O * (
                       M_FeO + M_FeSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (M_MgO * (
                       2.0 * M_SiO2 + 2.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                       4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (2.0 * M_SiO2 + 2.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                   M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                   M_SiO2 - M_m))))) + M_Si * (M_Fe * (
        M_FeSiO3 * (M_FeO * (M_SiO2 * (dM_MgO_er + dM_MgSiO3_er) + M_m * (-dM_MgO_er - dM_MgSiO3_er)) + M_MgO * (
        M_SiO2 * (1.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
        1.0 * dM_FeO_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                    -dM_MgO_er - dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
        M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
        1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er + 1.0 * dM_SiO2_er)) + M_MgO * (
                                                              M_SiO2 * (-1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er) + M_m * (
                                                              1.0 * dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                              1.0 * dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_O * (
        M_FeO * (M_SiO2 * (dM_MgO_er + dM_MgSiO3_er) + M_m * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
        M_FeO * (9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_MgO * (
        6.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 15.0 * dM_MgO_er + 24.0 * dM_MgSiO3_er + 9.0 * dM_SiO2_er) + M_SiO2 * (
        4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_m * (-25.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
        2 * dM_FeO_er + 5.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                                                                                                            4.0 * dM_FeO_er + 10.0 * dM_FeSiO3_er + 6.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_MgSiO3 * (
        M_FeO * (
        -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_MgO * (
        9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
        -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 6.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
        10.0 * dM_FeO_er + 25.0 * dM_FeSiO3_er - 15.0 * dM_MgO_er + 15.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
        -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_SiO2 * M_m * (
        M_FeO * (-dM_MgO_er - dM_MgSiO3_er) + M_MgO * (1.0 * dM_FeO_er + 1.0 * dM_FeSiO3_er))) + M_O * (M_FeO * (
        M_MgO * (M_SiO2 * (
        3.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                 6.0 * dM_FeSiO3_er + 6.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
        -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
        M_MgO * (9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
        9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_m * (-9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
        3.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er) + M_m * (
                                                                                                          -6.0 * dM_FeO_er + 6.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                              -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                                                                                                        M_FeO * (
                                                                                                        M_MgO * (
                                                                                                        9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                        -6.0 * dM_FeO_er - 12.0 * dM_FeSiO3_er - 6.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
                                                                                                        15.0 * dM_FeSiO3_er - 15.0 * dM_MgO_er + 15.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                        M_FeO * (
                                                                                                        9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                        9.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                        -6.0 * dM_FeSiO3_er - 6.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_m * (
                                                                                                        -15.0 * dM_FeO_er - 15.0 * dM_MgO_er + 15.0 * dM_SiO2_er)))) + dKSiO2_KSiO2 * (
                                                                                                               M_O * (
                                                                                                               M_Fe * (
                                                                                                               M_MgO * (
                                                                                                               M_FeSiO3 * (
                                                                                                               4.0 * M_SiO2 - 10.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               2.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                                               6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgO * (
                                                                                                               -6.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               M_MgO * (
                                                                                                               6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               6.0 * M_SiO2 - 6.0 * M_m) + M_MgO * (
                                                                                                               6.0 * M_SiO2 - 6.0 * M_m) - 6.0 * M_SiO2 * M_m))) + M_c * (
                                                                                                               M_Fe * (
                                                                                                               M_MgO * (
                                                                                                               M_FeSiO3 * (
                                                                                                               -M_SiO2 + 3.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               -M_SiO2 - M_m) + M_MgO * (
                                                                                                               -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_O * (
                                                                                                               M_MgO * (
                                                                                                               -M_FeSiO3 - M_SiO2) + M_MgSiO3 * (
                                                                                                               M_FeO - M_m))) + M_MgO * (
                                                                                                               2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               M_MgO * (
                                                                                                               -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                                               -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_O * (
                                                                                                               M_MgO * M_SiO2 * (
                                                                                                               -M_FeO - M_FeSiO3) + M_MgSiO3 * M_m * (
                                                                                                               -M_FeO - M_FeSiO3))))) + M_c * (
                       M_Fe * (M_FeSiO3 * (
                       M_FeO * (M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_MgO * (
                       M_SiO2 * (-dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                       -dM_FeO_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                       dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
                       M_SiO2 * (dM_FeO_er + 2.0 * dM_FeSiO3_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_m * (
                       -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er)) + M_MgO * (
                                                                M_SiO2 * (dM_FeO_er + 1.0 * dM_FeSiO3_er) + M_m * (
                                                                -dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                -dM_FeO_er - 1.0 * dM_FeSiO3_er)) + M_O * (M_FeSiO3 * (
                       M_FeO * (-dM_MgO_er - dM_MgSiO3_er) + M_MgO * (
                       -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                       dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (
                       dM_FeSiO3_er - dM_MgO_er + dM_SiO2_er) + M_MgO * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er)) + M_SiO2 * (
                                                                                                           M_FeO * (
                                                                                                           -dM_MgO_er - dM_MgSiO3_er) + M_MgO * (
                                                                                                           -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                           dM_MgO_er + dM_MgSiO3_er))) + M_SiO2 * M_m * (
                               M_FeO * (dM_MgO_er + dM_MgSiO3_er) + M_MgO * (
                               -dM_FeO_er - 1.0 * dM_FeSiO3_er))) + M_O * (M_FeO * M_SiO2 * (
                       M_MgO * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                       dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (
                       M_MgO * (-dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                       -dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_SiO2 * (M_MgO * (
                       -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                  dM_MgO_er + dM_MgSiO3_er))) + M_MgSiO3 * (
                                                                           M_FeO * (M_MgO * (
                                                                           -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                                    -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er)) + M_FeSiO3 * (
                                                                           M_FeO * (
                                                                           -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_MgO * (
                                                                           -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_m * (
                                                                           dM_FeO_er + dM_MgO_er - dM_SiO2_er)))) + M_Si * (
                       M_Fe * (
                       M_FeO * (M_SiO2 * (-dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (
                       M_FeO * (-4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                       -2.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_SiO2 * (
                       -dM_MgO_er - dM_MgSiO3_er) + M_m * (9.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er)) + M_MgO * (M_SiO2 * (
                       -dM_FeO_er - 3.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                                             -dM_FeO_er - 3.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_MgSiO3 * (
                       M_FeO * (
                       2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_MgO * (
                       -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                       dM_FeO_er + 3.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                       -3.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er - 6.0 * dM_SiO2_er)) + M_O * (
                       M_FeO * (-dM_MgO_er - dM_MgSiO3_er) + M_FeSiO3 * (-dM_MgO_er - dM_MgSiO3_er) + M_MgO * (
                       -dM_MgO_er - dM_MgSiO3_er) + M_MgSiO3 * (-dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                       -dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_SiO2 * M_m * (
                       4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_FeO * (M_MgO * (M_SiO2 * (
                       -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                  -2.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                         4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
                       M_FeO * (M_MgO * (
                       -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                -4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_m * (
                                4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_MgO * (
                       M_SiO2 * (-2.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                       2.0 * dM_FeO_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                       4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er)) + M_MgSiO3 * (M_FeO * (M_MgO * (
                       -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                     2 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                     -6.0 * dM_FeSiO3_er + 6.0 * dM_MgO_er - 6.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                            M_FeO * (
                                                                            -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                                            -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                            2.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                            6.0 * dM_FeO_er + 6.0 * dM_MgO_er - 6.0 * dM_SiO2_er))) + M_O * (
                       M_FeO * (M_MgO * (-dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                       -dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_FeSiO3 * (
                       M_MgO * (-dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_SiO2 * (
                       -dM_MgO_er - dM_MgSiO3_er) + M_m * (dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (
                       M_FeO * (-dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er) + M_FeSiO3 * (
                       -dM_FeO_er - dM_FeSiO3_er - dM_MgO_er - dM_MgSiO3_er))))) + dKMgO_KMgO * (M_Fe * M_O * (M_MgO * (
        4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
        M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
        -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Si * (M_Fe * (
        M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
        M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
        M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
        M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
        -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
        -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                      -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                      -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_O * (
                                                                                                           M_MgO * (
                                                                                                           9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           M_MgO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                           M_FeO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
                                                                                                           -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
                                                                                                 M_Fe * (M_MgO * (
                                                                                                 -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                         M_FeO * (
                                                                                                         M_MgO * (
                                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                         M_FeO * (
                                                                                                         M_SiO2 - M_m) + M_MgO * (
                                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                                                                         M_MgO * (
                                                                                                         M_FeSiO3 * (
                                                                                                         M_FeO - M_m) + M_SiO2 * (
                                                                                                         M_FeO - M_m)) + M_MgSiO3 * (
                                                                                                         M_FeO * (
                                                                                                         M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                                         M_FeO + M_MgO - M_m) + M_MgO * (
                                                                                                         M_SiO2 - M_m) - M_SiO2 * M_m))) + M_O * (
                                                                                                 M_MgO * (
                                                                                                 -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (M_MgO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 M_SiO2 - M_m) + M_MgO * (
                                                                                                 M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (
                                                                                                 M_Fe * (M_MgO * (
                                                                                                 M_FeO * (
                                                                                                 M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                 4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                         M_FeO * (
                                                                                                         4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                         4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                         4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                         M_MgO * (
                                                                                                         M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                         M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_MgO * (
                                                                                                 -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (M_MgO * (
                                                                                                 4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                                 4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                                 M_MgO * (M_FeO * (
                                                                                                 M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                          M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                 M_SiO2 - M_m))))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_m_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_m * (M_Fe * (M_O * (M_MgO * (M_FeO * M_SiO2 * (
        -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_FeSiO3 * (
                                              M_FeO * (
                                              -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                              -4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                     M_FeO * (M_MgO * (
                                     -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                              -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                     M_FeO * (
                                     -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                     -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                     -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)))) + dKFeO_KFeO * (
                              M_Mg * M_O * M_SiO2 * (-4.0 * M_FeO * M_MgSiO3 + 4.0 * M_FeSiO3 * M_MgO) + M_Si * (
                              M_Mg * (M_O * (M_FeO * (-15.0 * M_MgSiO3 - 3.0 * M_SiO2) + M_FeSiO3 * (
                              -9.0 * M_FeO + 6.0 * M_MgO + 6.0 * M_SiO2)) + M_SiO2 * (
                                      -M_FeO * M_MgSiO3 + M_FeSiO3 * M_MgO)) + M_O * (
                              M_MgO * (-3.0 * M_FeO * M_SiO2 + M_FeSiO3 * (-9.0 * M_FeO + 6.0 * M_SiO2)) + M_MgSiO3 * (
                              M_FeO * (-9.0 * M_MgO - 3.0 * M_SiO2) + M_FeSiO3 * (
                              -9.0 * M_FeO - 9.0 * M_MgO + 6.0 * M_SiO2)))) + M_c * (M_Mg * (M_FeO * (
                              M_MgSiO3 * M_SiO2 + M_O * (
                              M_FeSiO3 + M_MgSiO3 + M_SiO2)) - M_FeSiO3 * M_MgO * M_SiO2) + M_O * (M_FeO * M_MgO * (
                              M_FeSiO3 + M_SiO2) + M_MgSiO3 * (M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (
                              M_FeO + M_MgO))) + M_Si * (M_Mg * (M_FeO * (6.0 * M_MgSiO3 + 2.0 * M_SiO2) + M_FeSiO3 * (
                              4.0 * M_FeO - 2.0 * M_MgO - 2.0 * M_SiO2) + M_O * (M_FeO + M_FeSiO3)) + M_MgO * (
                                                         2.0 * M_FeO * M_SiO2 + M_FeSiO3 * (
                                                         4.0 * M_FeO - 2.0 * M_SiO2)) + M_MgSiO3 * (
                                                         M_FeO * (4.0 * M_MgO + 2.0 * M_SiO2) + M_FeSiO3 * (
                                                         4.0 * M_FeO + 4.0 * M_MgO - 2.0 * M_SiO2)) + M_O * (
                                                         M_MgO * (M_FeO + M_FeSiO3) + M_MgSiO3 * (
                                                         M_FeO + M_FeSiO3)))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                      M_O * M_SiO2 * (M_Fe * M_FeO * (4.0 * M_MgO + 4.0 * M_MgSiO3) + M_Mg * (
                      M_Fe * (4.0 * M_FeO + 4.0 * M_MgO) + M_FeO * (4.0 * M_MgO + 4.0 * M_MgSiO3))) + M_Si * (M_Fe * (
                      M_FeO * M_SiO2 * (M_MgO + M_MgSiO3) + M_O * (M_MgO * (6.0 * M_FeO + 6.0 * M_SiO2) + M_MgSiO3 * (
                      6.0 * M_FeO - 9.0 * M_MgO + 6.0 * M_SiO2))) + M_FeO * M_O * M_SiO2 * (
                                                                                                              9.0 * M_MgO + 9.0 * M_MgSiO3) + M_Mg * (
                                                                                                              M_Fe * (
                                                                                                              M_O * (
                                                                                                              6.0 * M_FeO + 6.0 * M_MgO + 6.0 * M_SiO2) + M_SiO2 * (
                                                                                                              M_FeO + M_MgO)) + M_FeO * (
                                                                                                              M_O * (
                                                                                                              6.0 * M_MgO + 15.0 * M_MgSiO3 + 9.0 * M_SiO2) + M_SiO2 * (
                                                                                                              M_MgO + M_MgSiO3)))) + M_c * (
                      M_Fe * (
                      -M_FeO * M_MgO * M_SiO2 + M_MgSiO3 * (-M_FeO * M_SiO2 + M_MgO * M_O)) + M_FeO * M_O * M_SiO2 * (
                      -M_MgO - M_MgSiO3) + M_Mg * (M_Fe * M_SiO2 * (-M_FeO - M_MgO) + M_FeO * (
                      M_O * (-M_MgSiO3 - M_SiO2) + M_SiO2 * (-M_MgO - M_MgSiO3))) + M_Si * (M_Fe * (
                      M_MgO * (-2.0 * M_FeO - 2.0 * M_SiO2) + M_MgSiO3 * (
                      -2.0 * M_FeO + 4.0 * M_MgO - 2.0 * M_SiO2) + M_O * (M_MgO + M_MgSiO3)) + M_FeO * M_SiO2 * (
                                                                                            -4.0 * M_MgO - 4.0 * M_MgSiO3) + M_Mg * (
                                                                                            M_Fe * (
                                                                                            -2.0 * M_FeO - 2.0 * M_MgO + M_O - 2.0 * M_SiO2) + M_FeO * (
                                                                                            -2.0 * M_MgO - 6.0 * M_MgSiO3 - 4.0 * M_SiO2))))) + M_Mg * (
                      M_O * (M_Fe * (M_FeSiO3 * (
                      M_FeO * (-4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                      -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                      -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_MgSiO3 * (M_FeO * (
                      -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                  -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_SiO2 * (
                                     M_FeO * (
                                     -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_MgO * (
                                     -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er))) + M_MgO * (
                             M_FeO * M_SiO2 * (
                             -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er) + M_FeSiO3 * (
                             M_FeO * (
                             -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                             -4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er))) + M_MgSiO3 * (
                             M_FeO * (M_MgO * (
                             -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                      -4.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                             M_FeO * (
                             -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_MgO * (
                             -4.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                             -4.0 * dM_FeSiO3_er - 4.0 * dM_MgSiO3_er - 4.0 * dM_SiO2_er)))) + dKMgO_KMgO * (
                      M_Fe * M_O * M_SiO2 * (4.0 * M_FeO * M_MgSiO3 - 4.0 * M_FeSiO3 * M_MgO) + M_Si * (M_Fe * (M_O * (
                      M_MgO * (-15.0 * M_FeSiO3 - 3.0 * M_SiO2) + M_MgSiO3 * (
                      6.0 * M_FeO - 9.0 * M_MgO + 6.0 * M_SiO2)) + M_SiO2 * (
                                                                                                                M_FeO * M_MgSiO3 - M_FeSiO3 * M_MgO)) + M_O * (
                                                                                                        M_MgO * (
                                                                                                        -3.0 * M_FeO * M_SiO2 + M_FeSiO3 * (
                                                                                                        -9.0 * M_FeO - 3.0 * M_SiO2)) + M_MgSiO3 * (
                                                                                                        M_FeO * (
                                                                                                        -9.0 * M_MgO + 6.0 * M_SiO2) + M_FeSiO3 * (
                                                                                                        -9.0 * M_FeO - 9.0 * M_MgO + 6.0 * M_SiO2)))) + M_c * (
                      M_Fe * (-M_FeO * M_MgSiO3 * M_SiO2 + M_MgO * (
                      M_FeSiO3 * M_SiO2 + M_O * (M_FeSiO3 + M_MgSiO3 + M_SiO2))) + M_O * (
                      M_MgO * (M_FeO * M_SiO2 + M_FeSiO3 * (M_FeO + M_SiO2)) + M_MgSiO3 * (
                      M_FeO * M_MgO + M_FeSiO3 * (M_FeO + M_MgO))) + M_Si * (M_Fe * (
                      M_MgO * (6.0 * M_FeSiO3 + 2.0 * M_SiO2) + M_MgSiO3 * (
                      -2.0 * M_FeO + 4.0 * M_MgO - 2.0 * M_SiO2) + M_O * (M_MgO + M_MgSiO3)) + M_MgO * (
                                                                             2.0 * M_FeO * M_SiO2 + M_FeSiO3 * (
                                                                             4.0 * M_FeO + 2.0 * M_SiO2)) + M_MgSiO3 * (
                                                                             M_FeO * (
                                                                             4.0 * M_MgO - 2.0 * M_SiO2) + M_FeSiO3 * (
                                                                             4.0 * M_FeO + 4.0 * M_MgO - 2.0 * M_SiO2)) + M_O * (
                                                                             M_MgO * (M_FeO + M_FeSiO3) + M_MgSiO3 * (
                                                                             M_FeO + M_FeSiO3)))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                      M_O * M_SiO2 * (M_Fe * M_MgO * (4.0 * M_FeO + 4.0 * M_FeSiO3) + M_Mg * (
                      M_Fe * (4.0 * M_FeO + 4.0 * M_MgO) + M_MgO * (4.0 * M_FeO + 4.0 * M_FeSiO3))) + M_Si * (M_Mg * (
                      M_Fe * (
                      M_O * (6.0 * M_FeO + 6.0 * M_MgO + 6.0 * M_SiO2) + M_SiO2 * (M_FeO + M_MgO)) + M_MgO * M_SiO2 * (
                      M_FeO + M_FeSiO3) + M_O * (M_FeO * (6.0 * M_MgO + 6.0 * M_SiO2) + M_FeSiO3 * (
                      -9.0 * M_FeO + 6.0 * M_MgO + 6.0 * M_SiO2))) + M_MgO * (M_Fe * (
                      M_O * (6.0 * M_FeO + 15.0 * M_FeSiO3 + 9.0 * M_SiO2) + M_SiO2 * (
                      M_FeO + M_FeSiO3)) + M_O * M_SiO2 * (9.0 * M_FeO + 9.0 * M_FeSiO3))) + M_c * (M_Mg * (
                      M_FeSiO3 * (M_FeO * M_O - M_MgO * M_SiO2) + M_SiO2 * (
                      M_Fe * (-M_FeO - M_MgO) - M_FeO * M_MgO)) + M_MgO * (M_Fe * (
                      M_O * (-M_FeSiO3 - M_SiO2) + M_SiO2 * (-M_FeO - M_FeSiO3)) + M_O * M_SiO2 * (
                                                                           -M_FeO - M_FeSiO3)) + M_Si * (M_Mg * (
                      M_Fe * (-2.0 * M_FeO - 2.0 * M_MgO + M_O - 2.0 * M_SiO2) + M_FeO * (
                      -2.0 * M_MgO - 2.0 * M_SiO2) + M_FeSiO3 * (4.0 * M_FeO - 2.0 * M_MgO - 2.0 * M_SiO2) + M_O * (
                      M_FeO + M_FeSiO3)) + M_MgO * (M_Fe * (-2.0 * M_FeO - 6.0 * M_FeSiO3 - 4.0 * M_SiO2) + M_SiO2 * (
                      -4.0 * M_FeO - 4.0 * M_FeSiO3))))) + M_Si * (M_Fe * (M_MgO * (
        M_FeO * M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_FeSiO3 * (
        M_FeO * (-1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
        -1.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er))) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
        -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                  -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                  -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er))) + M_O * (
                                                                           M_MgO * (M_FeO * (
                                                                           -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                    -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 25.0 * dM_MgO_er - 40.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                    -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 18.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                           M_FeO * (
                                                                           -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                           -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er) + M_MgO * (
                                                                           -9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                           -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)))) + M_Mg * (
                                                                   M_Fe * (M_FeSiO3 * (M_FeO * (
                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                       -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                       -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                           M_FeO * (
                                                                           -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                           -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                           -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_O * (
                                                                           M_FeO * (
                                                                           -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                           -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er) + M_MgO * (
                                                                           -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                           -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er) + M_SiO2 * (
                                                                           -6.0 * dM_FeO_er - 15.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_SiO2 * (
                                                                           M_FeO * (
                                                                           -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_MgO * (
                                                                           -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er))) + M_MgO * (
                                                                   M_FeO * M_SiO2 * (
                                                                   -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_FeSiO3 * (
                                                                   M_FeO * (
                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                   -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er))) + M_MgSiO3 * (
                                                                   M_FeO * (M_MgO * (
                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                            -dM_FeO_er - 2.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                   M_FeO * (
                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                   -1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                   -1.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er))) + M_O * (
                                                                   M_FeO * (M_MgO * (
                                                                   -4.0 * dM_FeO_er - 10.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_SiO2 * (
                                                                            -9.0 * dM_FeO_er - 18.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                   M_FeO * (
                                                                   -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er) + M_MgO * (
                                                                   2.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 4.0 * dM_MgO_er - 10.0 * dM_MgSiO3_er - 6.0 * dM_SiO2_er) + M_SiO2 * (
                                                                   -9.0 * dM_FeSiO3_er - 6.0 * dM_MgO_er - 15.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                   M_FeO * (
                                                                   -25.0 * dM_FeO_er - 40.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                   -10.0 * dM_FeO_er - 25.0 * dM_FeSiO3_er - 10.0 * dM_MgO_er - 25.0 * dM_MgSiO3_er - 15.0 * dM_SiO2_er)))) + M_O * (
                                                                   M_MgO * (M_FeO * M_SiO2 * (
                                                                   -9.0 * dM_FeO_er - 18.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 18.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                            M_FeO * (
                                                                            -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                            -9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 18.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                                                   M_FeO * (M_MgO * (
                                                                   -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                            -9.0 * dM_FeO_er - 18.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                   M_FeO * (
                                                                   -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_MgO * (
                                                                   -9.0 * dM_FeO_er - 9.0 * dM_FeSiO3_er - 9.0 * dM_MgO_er - 9.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                   -9.0 * dM_FeSiO3_er - 9.0 * dM_MgSiO3_er - 9.0 * dM_SiO2_er)))) + dKSiO2_KSiO2 * (
                                                                   M_O * (M_Fe * (M_MgO * (
                                                                   2.0 * M_FeO * M_SiO2 + M_FeSiO3 * (
                                                                   6.0 * M_FeO - 4.0 * M_SiO2)) + M_MgSiO3 * (M_FeO * (
                                                                   6.0 * M_MgO + 2.0 * M_SiO2) + M_FeSiO3 * (
                                                                                                              6.0 * M_FeO + 6.0 * M_MgO - 4.0 * M_SiO2))) + M_Mg * (
                                                                          M_Fe * (M_FeSiO3 * (
                                                                          6.0 * M_FeO + 6.0 * M_MgO - 4.0 * M_SiO2) + M_MgSiO3 * (
                                                                                  6.0 * M_FeO + 6.0 * M_MgO - 4.0 * M_SiO2) + M_SiO2 * (
                                                                                  2.0 * M_FeO + 2.0 * M_MgO)) + M_MgO * (
                                                                          2.0 * M_FeO * M_SiO2 + M_FeSiO3 * (
                                                                          6.0 * M_FeO + 2.0 * M_SiO2)) + M_MgSiO3 * (
                                                                          M_FeO * (
                                                                          6.0 * M_MgO - 4.0 * M_SiO2) + M_FeSiO3 * (
                                                                          6.0 * M_FeO + 6.0 * M_MgO - 4.0 * M_SiO2)))) + M_c * (
                                                                   M_Fe * (M_MgO * (-M_FeO * M_SiO2 + M_FeSiO3 * (
                                                                   -2.0 * M_FeO + M_SiO2)) + M_MgSiO3 * (M_FeO * (
                                                                   -2.0 * M_MgO - M_SiO2) + M_FeSiO3 * (
                                                                                                         -2.0 * M_FeO - 2.0 * M_MgO + M_SiO2)) + M_O * (
                                                                           M_MgO * (M_FeSiO3 + M_SiO2) + M_MgSiO3 * (
                                                                           M_FeSiO3 + M_SiO2))) + M_Mg * (M_Fe * (
                                                                   M_FeSiO3 * (
                                                                   -2.0 * M_FeO - 2.0 * M_MgO + M_SiO2) + M_MgSiO3 * (
                                                                   -2.0 * M_FeO - 2.0 * M_MgO + M_SiO2) + M_O * (
                                                                   M_FeSiO3 + M_MgSiO3 + M_SiO2) + M_SiO2 * (
                                                                   -M_FeO - M_MgO)) + M_MgO * (
                                                                                                          -M_FeO * M_SiO2 + M_FeSiO3 * (
                                                                                                          -2.0 * M_FeO - M_SiO2)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          -2.0 * M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                                          -2.0 * M_FeO - 2.0 * M_MgO + M_SiO2)) + M_O * (
                                                                                                          M_MgSiO3 * (
                                                                                                          M_FeO + M_FeSiO3) + M_SiO2 * (
                                                                                                          M_FeO + M_FeSiO3))) + M_O * M_SiO2 * (
                                                                   M_MgO * (M_FeO + M_FeSiO3) + M_MgSiO3 * (
                                                                   M_FeO + M_FeSiO3))))) + M_c * (M_Fe * (M_MgO * (
        M_FeO * M_SiO2 * (
        dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_FeSiO3 * (
        M_FeO * (dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
        dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er))) + M_MgSiO3 * (M_FeO * (
        M_MgO * (dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
        dM_FeO_er + 2.0 * dM_FeSiO3_er + 1.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                               dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er))) + M_O * (
                                                                                                          M_MgO * (
                                                                                                          M_FeSiO3 * (
                                                                                                          dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_SiO2 * (
                                                                                                          dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                          M_FeSiO3 * (
                                                                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_MgO * (
                                                                                                          dM_MgO_er + dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)))) + M_Mg * (
                                                                                                  M_Fe * (M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                  dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                          dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_O * (
                                                                                                          M_FeSiO3 * (
                                                                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_MgSiO3 * (
                                                                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_SiO2 * (
                                                                                                          dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_SiO2 * (
                                                                                                          M_FeO * (
                                                                                                          dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_MgO * (
                                                                                                          dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er))) + M_MgO * (
                                                                                                  M_FeO * M_SiO2 * (
                                                                                                  dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  1.0 * dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + 1.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                                                                                  M_FeO * (M_MgO * (
                                                                                                  dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                           dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                  dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er))) + M_O * (
                                                                                                  M_FeO * M_SiO2 * (
                                                                                                  dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er) + M_SiO2 * (
                                                                                                  dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)))) + M_O * (
                                                                                                  M_MgO * (
                                                                                                  M_FeO * M_SiO2 * (
                                                                                                  dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  dM_FeSiO3_er + dM_MgO_er + 2 * dM_MgSiO3_er + dM_SiO2_er))) + M_MgSiO3 * (
                                                                                                  M_FeO * (M_MgO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_SiO2 * (
                                                                                                           dM_FeO_er + 2 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_MgO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er)))) + M_Si * (
                                                                                                  M_Fe * (M_MgO * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 9.0 * dM_MgO_er + 15.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                  2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          dM_FeO_er + 3.0 * dM_FeSiO3_er - 1.0 * dM_MgO_er + 1.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                          3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_MgO * (
                                                                                                          4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_O * (
                                                                                                          M_MgO * (
                                                                                                          dM_MgO_er + dM_MgSiO3_er) + M_MgSiO3 * (
                                                                                                          dM_MgO_er + dM_MgSiO3_er))) + M_Mg * (
                                                                                                  M_Fe * (M_FeO * (
                                                                                                  dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                          3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_MgO * (
                                                                                                          dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                                                          3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                          2.0 * dM_FeO_er + 6.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_FeO * (
                                                                                                  M_MgO * (
                                                                                                  dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                  4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er) + M_MgO * (
                                                                                                  -1.0 * dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                  4.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 6.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  9.0 * dM_FeO_er + 15.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  3.0 * dM_FeO_er + 9.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 9.0 * dM_MgSiO3_er + 6.0 * dM_SiO2_er)) + M_O * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er) + M_FeSiO3 * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er))) + M_MgO * (
                                                                                                  M_FeO * M_SiO2 * (
                                                                                                  4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 8.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                                                                                  M_FeO * (M_MgO * (
                                                                                                  4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                           4.0 * dM_FeO_er + 8.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                  4.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 4.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  4.0 * dM_FeSiO3_er + 4.0 * dM_MgSiO3_er + 4.0 * dM_SiO2_er))) + M_O * (
                                                                                                  M_MgO * (M_FeO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_FeSiO3 * (
                                                                                                           dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er)) + M_MgSiO3 * (
                                                                                                  M_FeO * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er) + M_FeSiO3 * (
                                                                                                  dM_FeO_er + dM_FeSiO3_er + dM_MgO_er + dM_MgSiO3_er)))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_c_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_c * (M_Fe * (M_O * (M_MgO * (M_FeSiO3 * (
        M_FeO * (-2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
        2.0 * dM_FeO_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_SiO2 * (M_FeO * (
        -2.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                              2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er))) + M_MgSiO3 * (
                                     M_FeO * (M_MgO * (
                                     -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                              -2.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                     M_FeO * (
                                     -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                                     -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
                                     2.0 * dM_FeO_er + 2.0 * dM_MgO_er - 2.0 * dM_SiO2_er)) + M_MgO * (
                                     M_SiO2 * (-2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er) + M_m * (
                                     2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                     2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er))) + dKFeO_KFeO * (M_O * (M_Mg * (
        M_FeO * M_m * (-2.0 * M_MgSiO3 - 2.0 * M_SiO2) + M_FeSiO3 * (
        M_FeO * (2.0 * M_SiO2 - 2.0 * M_m) + M_SiO2 * (2.0 * M_MgO - 2.0 * M_m))) + M_MgO * (
                                                                                                    -2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                    M_FeO * (
                                                                                                    2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                    M_FeO * (M_MgO * (
                                                                                                    2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                    M_FeO * (
                                                                                                    2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (
                                                                                                    2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m))) + M_Si * (
                                                                                             M_Mg * (M_FeO * (
                                                                                             M_MgSiO3 * (
                                                                                             M_SiO2 - 3.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                     M_FeO * (
                                                                                                     2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (
                                                                                                     M_SiO2 + M_m) - 2.0 * M_SiO2 * M_m) + M_O * (
                                                                                                     M_FeO * (
                                                                                                     -3.0 * M_MgSiO3 - M_SiO2 - 2.0 * M_m) + M_FeSiO3 * (
                                                                                                     3.0 * M_MgO + 2.0 * M_SiO2 - 5.0 * M_m))) + M_MgO * (
                                                                                             -2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                             M_FeO * (
                                                                                             2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                             M_FeO * (M_MgO * (
                                                                                             2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                             M_FeO * (
                                                                                             2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (
                                                                                             2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_O * (
                                                                                             M_MgO * (M_FeO * (
                                                                                             -M_SiO2 - 2.0 * M_m) + M_FeSiO3 * (
                                                                                                      2.0 * M_SiO2 - 5.0 * M_m)) + M_MgSiO3 * (
                                                                                             M_FeO * (
                                                                                             -M_SiO2 - 2.0 * M_m) + M_FeSiO3 * (
                                                                                             2.0 * M_SiO2 - 5.0 * M_m)))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                      M_O * (M_Fe * (M_MgO * M_SiO2 * (2.0 * M_FeO - 2.0 * M_m) + M_MgSiO3 * (
                      M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) + M_SiO2 * (2.0 * M_FeO - 2.0 * M_m))) + M_Mg * (
                             M_Fe * M_SiO2 * (2.0 * M_FeO + 2.0 * M_MgO - 2.0 * M_m) + M_FeO * (
                             2.0 * M_MgO * M_SiO2 + 2.0 * M_MgSiO3 * M_m))) + M_Si * (M_Fe * (
                      M_MgO * (M_FeO * (M_SiO2 + M_m) - 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                      M_FeO * (M_SiO2 + M_m) + M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_O * (
                      M_MgO * (3.0 * M_FeO + 2.0 * M_SiO2 - 5.0 * M_m) + M_MgSiO3 * (
                      3.0 * M_FeO + 2.0 * M_SiO2 - 5.0 * M_m))) + M_FeO * M_O * (M_MgO * (
                      3.0 * M_SiO2 - 3.0 * M_m) + M_MgSiO3 * (3.0 * M_SiO2 - 3.0 * M_m)) + M_Mg * (M_Fe * (
                      M_FeO * (M_SiO2 + M_m) + M_MgO * (M_SiO2 + M_m) + M_O * (
                      3.0 * M_FeO + 3.0 * M_MgO + 2.0 * M_SiO2 - 5.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeO * (M_MgO * (
                      M_SiO2 + M_m) + M_MgSiO3 * (-M_SiO2 + 3.0 * M_m) + M_O * (
                                                                                                             3.0 * M_MgO + 3.0 * M_MgSiO3 + 3.0 * M_SiO2 - 3.0 * M_m))))) + M_Mg * (
                      M_O * (M_Fe * (M_FeSiO3 * (
                      M_FeO * (-2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                      -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
                      2.0 * dM_FeO_er + 2.0 * dM_MgO_er - 2.0 * dM_SiO2_er)) + M_MgSiO3 * (M_FeO * (
                      -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                           -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
                                                                                           2.0 * dM_FeO_er + 2.0 * dM_MgO_er - 2.0 * dM_SiO2_er)) + M_SiO2 * (
                                     M_FeO * (
                                     -2.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_MgO * (
                                     -2.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                     2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er))) + M_FeO * M_SiO2 * (
                             M_MgO * (
                             -2.0 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                             2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (M_MgO * (
                      -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                           -2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
                                                                                           2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er)) + M_SiO2 * (
                                                                                  M_MgO * (
                                                                                  -2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                  2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er))) + M_MgSiO3 * (
                             M_FeO * (M_MgO * (
                             -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
                                      -2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
                             -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                               -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
                                                                                                               2.0 * dM_FeO_er + 2.0 * dM_MgO_er - 2.0 * dM_SiO2_er)))) + dKMgO_KMgO * (
                      M_O * (M_Fe * (M_MgO * M_m * (-2.0 * M_FeSiO3 - 2.0 * M_SiO2) + M_MgSiO3 * (
                      M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) + M_SiO2 * (2.0 * M_FeO - 2.0 * M_m))) + M_MgO * (
                             -2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                             M_FeO * (2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                             M_FeO * (M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                             M_FeO * (2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (
                             2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m))) + M_Si * (M_Fe * (
                      M_MgO * (M_FeSiO3 * (M_SiO2 - 3.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                      M_FeO * (M_SiO2 + M_m) + M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_O * (
                      M_MgO * (-3.0 * M_FeSiO3 - M_SiO2 - 2.0 * M_m) + M_MgSiO3 * (
                      3.0 * M_FeO + 2.0 * M_SiO2 - 5.0 * M_m))) + M_MgO * (-2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                      M_FeO * (2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
                      M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                      2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_O * (
                                                                                         M_MgO * (M_FeO * (
                                                                                         -M_SiO2 - 2.0 * M_m) + M_FeSiO3 * (
                                                                                                  -M_SiO2 - 2.0 * M_m)) + M_MgSiO3 * (
                                                                                         M_FeO * (
                                                                                         2.0 * M_SiO2 - 5.0 * M_m) + M_FeSiO3 * (
                                                                                         2.0 * M_SiO2 - 5.0 * M_m)))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                      M_O * (M_Fe * M_MgO * (2.0 * M_FeO * M_SiO2 + 2.0 * M_FeSiO3 * M_m) + M_Mg * (
                      M_FeSiO3 * (M_FeO * (2.0 * M_SiO2 - 2.0 * M_m) + M_SiO2 * (2.0 * M_MgO - 2.0 * M_m)) + M_SiO2 * (
                      M_Fe * (2.0 * M_FeO + 2.0 * M_MgO - 2.0 * M_m) + M_FeO * (2.0 * M_MgO - 2.0 * M_m)))) + M_Si * (
                      M_Mg * (M_Fe * (M_FeO * (M_SiO2 + M_m) + M_MgO * (M_SiO2 + M_m) + M_O * (
                      3.0 * M_FeO + 3.0 * M_MgO + 2.0 * M_SiO2 - 5.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeO * (
                              M_MgO * (M_SiO2 + M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (
                              M_SiO2 + M_m) - 2.0 * M_SiO2 * M_m) + M_O * (
                              M_FeO * (3.0 * M_MgO + 2.0 * M_SiO2 - 5.0 * M_m) + M_FeSiO3 * (
                              3.0 * M_MgO + 2.0 * M_SiO2 - 5.0 * M_m))) + M_MgO * (M_Fe * (
                      M_FeO * (M_SiO2 + M_m) + M_FeSiO3 * (-M_SiO2 + 3.0 * M_m) + M_O * (
                      3.0 * M_FeO + 3.0 * M_FeSiO3 + 3.0 * M_SiO2 - 3.0 * M_m)) + M_O * (M_FeO * (
                      3.0 * M_SiO2 - 3.0 * M_m) + M_FeSiO3 * (3.0 * M_SiO2 - 3.0 * M_m))))) + M_Si * (M_Fe * (M_MgO * (
        M_FeO * (M_SiO2 * (-dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
        -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er)) + M_FeSiO3 * (
        M_FeO * (-2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
        dM_FeSiO3_er + dM_MgO_er + 2.0 * dM_MgSiO3_er + dM_SiO2_er) + M_m * (
        3.0 * dM_FeO_er - 3.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
        2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (
        M_MgO * (-2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
        -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
        -dM_FeSiO3_er + dM_MgO_er - dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
        -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                               -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                               dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                               3.0 * dM_FeO_er + 3.0 * dM_MgO_er - 3.0 * dM_SiO2_er)) + M_MgO * (
                                                             M_SiO2 * (-2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er) + M_m * (
                                                             2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                             2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er)) + M_O * (M_MgO * (
        M_FeO * (
        -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_FeSiO3 * (
        -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 5.0 * dM_MgO_er - 8.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_SiO2 * (
        -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
        2.0 * dM_FeO_er + 5.0 * dM_FeSiO3_er + 3.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er)) + M_MgSiO3 * (M_FeO * (
        -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er + dM_MgO_er - 2.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                     -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                     -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 3.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                     2.0 * dM_FeO_er + 5.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er)))) + M_Mg * (
                                                                                                      M_Fe * (M_FeO * (
                                                                                                      M_SiO2 * (
                                                                                                      -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                      -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                              -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                              dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                              3.0 * dM_FeO_er + 3.0 * dM_MgO_er - 3.0 * dM_SiO2_er)) + M_MgO * (
                                                                                                              M_SiO2 * (
                                                                                                              -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                              -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                              -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                              dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                              3.0 * dM_FeO_er + 3.0 * dM_MgO_er - 3.0 * dM_SiO2_er)) + M_O * (
                                                                                                              M_FeO * (
                                                                                                              -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                              -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_MgO * (
                                                                                                              -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_MgSiO3 * (
                                                                                                              -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                              -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                              2.0 * dM_FeO_er + 5.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                              2.0 * dM_FeO_er + 2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er)) + M_FeO * (
                                                                                                      M_MgO * (
                                                                                                      M_SiO2 * (
                                                                                                      -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                      -dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                      2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                                                      M_FeO * (M_MgO * (
                                                                                                      -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                               -2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_m * (
                                                                                                               2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                                                      M_SiO2 * (
                                                                                                      -dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                      dM_FeO_er - dM_MgSiO3_er - dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                                                      2.0 * dM_MgO_er + 2.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                                                                                                      M_FeO * (M_MgO * (
                                                                                                      -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                               dM_FeO_er + 2.0 * dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                               -3.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er - 3.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                      -2.0 * dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 2.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                      dM_FeSiO3_er + dM_MgSiO3_er + dM_SiO2_er) + M_m * (
                                                                                                      3.0 * dM_FeO_er + 3.0 * dM_MgO_er - 3.0 * dM_SiO2_er))) + M_O * (
                                                                                                      M_FeO * (M_MgO * (
                                                                                                      -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                               -3.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                               3.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                      M_MgO * (
                                                                                                      dM_FeO_er - 2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                      -3.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                      -3.0 * dM_FeO_er + 2.0 * dM_MgO_er + 5.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      -5.0 * dM_FeO_er - 8.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                      -2.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er)))) + M_O * (
                                                                                                      M_MgO * (M_FeO * (
                                                                                                      M_SiO2 * (
                                                                                                      -3.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                      3.0 * dM_FeSiO3_er + 3.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                               M_SiO2 * (
                                                                                                               -3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 6.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                               -3.0 * dM_FeO_er + 3.0 * dM_MgSiO3_er + 3.0 * dM_SiO2_er))) + M_MgSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      M_SiO2 * (
                                                                                                      -3.0 * dM_FeO_er - 6.0 * dM_FeSiO3_er - 3.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                      3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                      M_SiO2 * (
                                                                                                      -3.0 * dM_FeSiO3_er - 3.0 * dM_MgSiO3_er - 3.0 * dM_SiO2_er) + M_m * (
                                                                                                      -3.0 * dM_FeO_er - 3.0 * dM_MgO_er + 3.0 * dM_SiO2_er)))) + dKSiO2_KSiO2 * (
                                                                                                      M_Fe * (M_MgO * (
                                                                                                      M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              M_MgO * (
                                                                                                              -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              -M_SiO2 + M_m) + M_MgO * (
                                                                                                              -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                                              M_MgO * (
                                                                                                              M_FeSiO3 * (
                                                                                                              3.0 * M_FeO - 5.0 * M_m) + M_SiO2 * (
                                                                                                              M_FeO - 3.0 * M_m)) + M_MgSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              3.0 * M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                                              3.0 * M_FeO + 3.0 * M_MgO - 5.0 * M_m) + M_MgO * (
                                                                                                              3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m))) + M_Mg * (
                                                                                                      M_Fe * (
                                                                                                      M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      -M_SiO2 + M_m) + M_MgO * (
                                                                                                      -M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      -M_SiO2 + M_m) + M_MgO * (
                                                                                                      -M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                                                                                                      M_FeSiO3 * (
                                                                                                      3.0 * M_FeO + 3.0 * M_MgO - 5.0 * M_m) + M_MgSiO3 * (
                                                                                                      3.0 * M_FeO + 3.0 * M_MgO - 5.0 * M_m) + M_SiO2 * (
                                                                                                      M_FeO + M_MgO - 3.0 * M_m)) + M_SiO2 * M_m * (
                                                                                                      M_FeO + M_MgO)) + M_MgO * (
                                                                                                      M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                      M_FeO * (M_MgO * (
                                                                                                      -M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      -M_SiO2 + M_m) + M_MgO * (
                                                                                                      -M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
                                                                                                      M_FeO * M_SiO2 * (
                                                                                                      M_MgO - 3.0 * M_m) + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      3.0 * M_MgO + 3.0 * M_SiO2 - 3.0 * M_m) + M_SiO2 * (
                                                                                                      M_MgO - 3.0 * M_m)) + M_MgSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      3.0 * M_MgO - 5.0 * M_m) + M_FeSiO3 * (
                                                                                                      3.0 * M_FeO + 3.0 * M_MgO - 5.0 * M_m)))) + M_O * (
                                                                                                      M_MgO * (
                                                                                                      -3.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                      M_FeO * (M_MgO * (
                                                                                                      3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                      M_FeO * (
                                                                                                      3.0 * M_SiO2 - 3.0 * M_m) + M_MgO * (
                                                                                                      3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m)))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_O_dTc(self, Moles, dKs, dMi_b):
        dM_Mg_er, dM_Si_er, dM_Fe_er, dM_O_er, dM_c_er, dM_MgO_er, dM_SiO2_er, dM_FeO_er, dM_MgSiO3_er, dM_FeSiO3_er, dM_m_er = self.unwrap_Moles(
            dMi_b)
        M_Mg, M_Si, M_Fe, M_O, M_c, M_MgO, M_SiO2, M_FeO, M_MgSiO3, M_FeSiO3, M_m = Moles
        dKMgO_KMgO, dKSiO2_KSiO2, dKFeO_KFeO, dKMgSiO3_KMgSiO3, dKFeSiO3_KFeSiO3 = dKs
        return M_O * (M_Fe * dKFeO_KFeO * (M_Si * (M_Mg * (
        M_FeO * (M_MgSiO3 * (2.0 * M_SiO2 - 5.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_FeSiO3 * (
        M_FeO * (3.0 * M_SiO2 - 3.0 * M_m) + M_MgO * (M_SiO2 + 2.0 * M_m) - 3.0 * M_SiO2 * M_m)) + M_MgO * (
                                                   -3.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (
                                                   3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                   M_FeO * (M_MgO * (
                                                   3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                   M_FeO * (3.0 * M_SiO2 - 3.0 * M_m) + M_MgO * (
                                                   3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m))) + M_c * (M_Mg * (
        M_FeO * M_m * (-M_MgSiO3 - M_SiO2) + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_SiO2 * (M_MgO - M_m))) + M_MgO * (
                                                                                                              -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              M_MgO * (
                                                                                                              M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              M_SiO2 - M_m) + M_MgO * (
                                                                                                              M_SiO2 - M_m) - M_SiO2 * M_m)) + M_Si * (
                                                                                                              M_Mg * (
                                                                                                              M_FeO * (
                                                                                                              -2.0 * M_MgSiO3 - M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                              2.0 * M_MgO + M_SiO2 - 3.0 * M_m)) + M_MgO * (
                                                                                                              M_FeO * (
                                                                                                              -M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                              M_SiO2 - 3.0 * M_m)) + M_MgSiO3 * (
                                                                                                              M_FeO * (
                                                                                                              -M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                              M_SiO2 - 3.0 * M_m))))) + M_FeSiO3 * dKFeSiO3_KFeSiO3 * (
                      M_Si * (M_Fe * (M_MgO * (M_FeO * (M_SiO2 + 2.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                      M_FeO * (M_SiO2 + 2.0 * M_m) + M_MgO * (
                      3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m)) + M_Mg * (M_Fe * (
                      M_FeO * (M_SiO2 + 2.0 * M_m) + M_MgO * (M_SiO2 + 2.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                 M_MgO * (
                                                                                 M_SiO2 + 2.0 * M_m) + M_MgSiO3 * (
                                                                                 -2.0 * M_SiO2 + 5.0 * M_m)))) + M_c * (
                      M_Fe * (M_MgO * M_SiO2 * (M_FeO - M_m) + M_MgSiO3 * (
                      M_MgO * (M_SiO2 - M_m) + M_SiO2 * (M_FeO - M_m))) + M_Mg * (
                      M_Fe * M_SiO2 * (M_FeO + M_MgO - M_m) + M_FeO * (M_MgO * M_SiO2 + M_MgSiO3 * M_m)) + M_Si * (
                      M_Fe * (M_MgO * (2.0 * M_FeO + M_SiO2 - 3.0 * M_m) + M_MgSiO3 * (
                      2.0 * M_FeO + M_SiO2 - 3.0 * M_m)) + M_FeO * (
                      M_MgO * (2.0 * M_SiO2 - 2.0 * M_m) + M_MgSiO3 * (2.0 * M_SiO2 - 2.0 * M_m)) + M_Mg * (
                      M_Fe * (2.0 * M_FeO + 2.0 * M_MgO + M_SiO2 - 3.0 * M_m) + M_FeO * (
                      2.0 * M_MgO + 2.0 * M_MgSiO3 + 2.0 * M_SiO2 - 2.0 * M_m))))) + M_Mg * dKMgO_KMgO * (M_Si * (
        M_Fe * (M_MgO * (M_FeSiO3 * (2.0 * M_SiO2 - 5.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_MgSiO3 * (
        M_FeO * (M_SiO2 + 2.0 * M_m) + M_MgO * (3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m)) + M_MgO * (
        -3.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
        M_FeO * (3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
        M_FeO * (M_MgO * (3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_FeSiO3 * (
        M_FeO * (3.0 * M_SiO2 - 3.0 * M_m) + M_MgO * (3.0 * M_SiO2 - 3.0 * M_m) - 3.0 * M_SiO2 * M_m))) + M_c * (
                                                                                                          M_Fe * (
                                                                                                          M_MgO * M_m * (
                                                                                                          -M_FeSiO3 - M_SiO2) + M_MgSiO3 * (
                                                                                                          M_MgO * (
                                                                                                          M_SiO2 - M_m) + M_SiO2 * (
                                                                                                          M_FeO - M_m))) + M_MgO * (
                                                                                                          -M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_MgO * (
                                                                                                          M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_SiO2 - M_m) + M_MgO * (
                                                                                                          M_SiO2 - M_m) - M_SiO2 * M_m)) + M_Si * (
                                                                                                          M_Fe * (
                                                                                                          M_MgO * (
                                                                                                          -2.0 * M_FeSiO3 - M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                          2.0 * M_FeO + M_SiO2 - 3.0 * M_m)) + M_MgO * (
                                                                                                          M_FeO * (
                                                                                                          -M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                          -M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                                          M_FeO * (
                                                                                                          M_SiO2 - 3.0 * M_m) + M_FeSiO3 * (
                                                                                                          M_SiO2 - 3.0 * M_m))))) + M_MgSiO3 * dKMgSiO3_KMgSiO3 * (
                      M_Si * (
                      M_Fe * M_MgO * (M_FeO * (M_SiO2 + 2.0 * M_m) + M_FeSiO3 * (-2.0 * M_SiO2 + 5.0 * M_m)) + M_Mg * (
                      M_Fe * (
                      M_FeO * (M_SiO2 + 2.0 * M_m) + M_MgO * (M_SiO2 + 2.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_FeO * (
                      M_MgO * (M_SiO2 + 2.0 * M_m) - 3.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                      M_FeO * (3.0 * M_SiO2 - 3.0 * M_m) + M_MgO * (
                      M_SiO2 + 2.0 * M_m) - 3.0 * M_SiO2 * M_m))) + M_c * (
                      M_Fe * M_MgO * (M_FeO * M_SiO2 + M_FeSiO3 * M_m) + M_Mg * (
                      M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_SiO2 * (M_MgO - M_m)) + M_SiO2 * (
                      M_Fe * (M_FeO + M_MgO - M_m) + M_FeO * (M_MgO - M_m))) + M_Si * (M_Mg * (
                      M_Fe * (2.0 * M_FeO + 2.0 * M_MgO + M_SiO2 - 3.0 * M_m) + M_FeO * (
                      2.0 * M_MgO + M_SiO2 - 3.0 * M_m) + M_FeSiO3 * (2.0 * M_MgO + M_SiO2 - 3.0 * M_m)) + M_MgO * (
                                                                                       M_Fe * (
                                                                                       2.0 * M_FeO + 2.0 * M_FeSiO3 + 2.0 * M_SiO2 - 2.0 * M_m) + M_FeO * (
                                                                                       2.0 * M_SiO2 - 2.0 * M_m) + M_FeSiO3 * (
                                                                                       2.0 * M_SiO2 - 2.0 * M_m))))) + M_Si * (
                      M_Fe * (M_MgO * (M_FeO * (
                      M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                      -2.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
                      -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er + 4.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                                  5.0 * dM_FeO_er - 5.0 * dM_MgSiO3_er - 5.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                       3.0 * dM_FeO_er + 3.0 * dM_FeSiO3_er)) + M_MgSiO3 * (M_FeO * (M_MgO * (
                      -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                     -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                                                     -2.0 * dM_FeSiO3_er + 2.0 * dM_MgO_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                            M_FeO * (
                                                                                            -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                            -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                            2.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                            5.0 * dM_FeO_er + 5.0 * dM_MgO_er - 5.0 * dM_SiO2_er)) + M_MgO * (
                                                                                            M_SiO2 * (
                                                                                            -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er) + M_m * (
                                                                                            3.0 * dM_FeO_er + 3.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                                                                                            3.0 * dM_FeO_er + 3.0 * dM_FeSiO3_er))) + M_Mg * (
                      M_Fe * (M_FeO * (
                      M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                      -2.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_FeSiO3 * (M_FeO * (
                      -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                  -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                  2.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                                  5.0 * dM_FeO_er + 5.0 * dM_MgO_er - 5.0 * dM_SiO2_er)) + M_MgO * (
                              M_SiO2 * (
                              -dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                              -2.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_MgSiO3 * (M_FeO * (
                      -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_MgO * (
                                                                                                          -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                                          2.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                                                                                                          5.0 * dM_FeO_er + 5.0 * dM_MgO_er - 5.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                              3.0 * dM_FeO_er + 3.0 * dM_FeSiO3_er + 3.0 * dM_MgO_er + 3.0 * dM_MgSiO3_er)) + M_FeO * (
                      M_MgO * (
                      M_SiO2 * (-dM_FeO_er - 2 * dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                      -2.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                      3.0 * dM_MgO_er + 3.0 * dM_MgSiO3_er)) + M_FeSiO3 * (M_FeO * (M_MgO * (
                      -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                    -3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_m * (
                                                                                    3.0 * dM_MgO_er + 3.0 * dM_MgSiO3_er)) + M_MgO * (
                                                                           M_SiO2 * (
                                                                           -dM_FeSiO3_er - dM_MgO_er - 2 * dM_MgSiO3_er - dM_SiO2_er) + M_m * (
                                                                           2.0 * dM_FeO_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er)) + M_SiO2 * M_m * (
                                                                           3.0 * dM_MgO_er + 3.0 * dM_MgSiO3_er)) + M_MgSiO3 * (
                      M_FeO * (M_MgO * (
                      -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                               2.0 * dM_FeO_er + 4.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                               -5.0 * dM_FeSiO3_er + 5.0 * dM_MgO_er - 5.0 * dM_SiO2_er)) + M_FeSiO3 * (
                      M_FeO * (-3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_MgO * (
                      -3.0 * dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 3.0 * dM_MgSiO3_er) + M_SiO2 * (
                      2.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er) + M_m * (
                      5.0 * dM_FeO_er + 5.0 * dM_MgO_er - 5.0 * dM_SiO2_er)))) + dKSiO2_KSiO2 * (M_Fe * (M_MgO * (
                      2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                      M_FeO * (-2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
                      M_MgO * (-2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
                      -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                              -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m))) + M_Mg * (
                                                                                                 M_Fe * (M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                                         M_FeO * (
                                                                                                         -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                                         -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                         2.0 * M_FeO + 2.0 * M_MgO)) + M_MgO * (
                                                                                                 2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (M_MgO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + M_MgO * (
                                                                                                 -2.0 * M_SiO2 + 2.0 * M_m) + 2.0 * M_SiO2 * M_m))) + M_c * (
                                                                                                 M_Fe * (M_MgO * (
                                                                                                 M_FeSiO3 * (
                                                                                                 2.0 * M_FeO - 3.0 * M_m) + M_SiO2 * (
                                                                                                 M_FeO - 2.0 * M_m)) + M_MgSiO3 * (
                                                                                                         M_FeO * (
                                                                                                         2.0 * M_MgO + M_SiO2) + M_FeSiO3 * (
                                                                                                         2.0 * M_FeO + 2.0 * M_MgO - 3.0 * M_m) + M_MgO * (
                                                                                                         2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_Mg * (
                                                                                                 M_Fe * (M_FeSiO3 * (
                                                                                                 2.0 * M_FeO + 2.0 * M_MgO - 3.0 * M_m) + M_MgSiO3 * (
                                                                                                         2.0 * M_FeO + 2.0 * M_MgO - 3.0 * M_m) + M_SiO2 * (
                                                                                                         M_FeO + M_MgO - 2.0 * M_m)) + M_FeO * M_SiO2 * (
                                                                                                 M_MgO - 2.0 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 2.0 * M_MgO + 2.0 * M_SiO2 - 2.0 * M_m) + M_SiO2 * (
                                                                                                 M_MgO - 2.0 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 2.0 * M_MgO - 3.0 * M_m) + M_FeSiO3 * (
                                                                                                 2.0 * M_FeO + 2.0 * M_MgO - 3.0 * M_m))) + M_MgO * (
                                                                                                 -2.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                                 M_FeO * (M_MgO * (
                                                                                                 2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                                 M_FeO * (
                                                                                                 2.0 * M_SiO2 - 2.0 * M_m) + M_MgO * (
                                                                                                 2.0 * M_SiO2 - 2.0 * M_m) - 2.0 * M_SiO2 * M_m))))) + M_c * (
                      M_Fe * (M_MgO * (M_FeSiO3 * (
                      M_FeO * (-dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                      dM_FeO_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_SiO2 * (M_FeO * (
                      -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                      dM_FeO_er + 1.0 * dM_FeSiO3_er))) + M_MgSiO3 * (
                              M_FeO * (
                              M_MgO * (-dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                              -dM_FeO_er - 2.0 * dM_FeSiO3_er - 1.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                              M_FeO * (-dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                              -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                              dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgO * (
                              M_SiO2 * (-dM_FeO_er - 1.0 * dM_FeSiO3_er) + M_m * (
                              dM_FeO_er + 1.0 * dM_FeSiO3_er)) + M_SiO2 * M_m * (
                              dM_FeO_er + 1.0 * dM_FeSiO3_er))) + M_Mg * (M_Fe * (M_FeSiO3 * (
                      M_FeO * (-dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                      -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                      dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_MgSiO3 * (M_FeO * (
                      -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                               -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                               dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_SiO2 * (
                                                                                  M_FeO * (
                                                                                  -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_MgO * (
                                                                                  -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                                  dM_FeO_er + 1.0 * dM_FeSiO3_er + dM_MgO_er + 1.0 * dM_MgSiO3_er))) + M_FeO * M_SiO2 * (
                                                                          M_MgO * (
                                                                          -dM_FeO_er - 2.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                          dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_FeSiO3 * (
                                                                          M_FeO * (M_MgO * (
                                                                          -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_SiO2 * (
                                                                                   -dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                   dM_MgO_er + 1.0 * dM_MgSiO3_er)) + M_SiO2 * (
                                                                          M_MgO * (
                                                                          -1.0 * dM_FeSiO3_er - dM_MgO_er - 2.0 * dM_MgSiO3_er - 1.0 * dM_SiO2_er) + M_m * (
                                                                          dM_MgO_er + 1.0 * dM_MgSiO3_er))) + M_MgSiO3 * (
                                                                          M_FeO * (M_MgO * (
                                                                          -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                                   -1.0 * dM_FeSiO3_er + dM_MgO_er - 1.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                          M_FeO * (
                                                                          -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_MgO * (
                                                                          -dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 1.0 * dM_MgSiO3_er) + M_m * (
                                                                          dM_FeO_er + dM_MgO_er - 1.0 * dM_SiO2_er)))) + M_Si * (
                      M_Fe * (M_MgO * (M_FeO * (
                      -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                       -dM_FeO_er - 3.0 * dM_FeSiO3_er - 3.0 * dM_MgO_er - 5.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_SiO2 * (
                                       -dM_FeO_er - 3.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                       dM_FeO_er + 3.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_MgSiO3 * (
                              M_FeO * (
                              -dM_FeO_er - 3.0 * dM_FeSiO3_er + 1.0 * dM_MgO_er - 1.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                              -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_SiO2 * (
                              -dM_FeO_er - 3.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                              dM_FeO_er + 3.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er + 2.0 * dM_SiO2_er))) + M_Mg * (M_Fe * (
                      M_FeO * (
                      -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                      -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_MgO * (
                      -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_MgSiO3 * (
                      -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_SiO2 * (
                      -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                      dM_FeO_er + 3.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_FeO * (
                                                                                                               M_MgO * (
                                                                                                               -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                               -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                                               2.0 * dM_FeSiO3_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                                                                                                               M_MgO * (
                                                                                                               1.0 * dM_FeO_er - 1.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_SiO2 * (
                                                                                                               -2.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                                               -2.0 * dM_FeO_er + dM_MgO_er + 3.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_MgSiO3 * (
                                                                                                               M_FeO * (
                                                                                                               -3.0 * dM_FeO_er - 5.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_FeSiO3 * (
                                                                                                               -dM_FeO_er - 3.0 * dM_FeSiO3_er - dM_MgO_er - 3.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er))) + M_MgO * (
                      M_FeO * (M_SiO2 * (
                      -2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                               2.0 * dM_FeSiO3_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er)) + M_FeSiO3 * (
                      M_SiO2 * (-2.0 * dM_FeSiO3_er - 2 * dM_MgO_er - 4.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                      -2.0 * dM_FeO_er + 2.0 * dM_MgSiO3_er + 2.0 * dM_SiO2_er))) + M_MgSiO3 * (M_FeO * (
                      M_SiO2 * (-2 * dM_FeO_er - 4.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                      2.0 * dM_FeSiO3_er - 2.0 * dM_MgO_er + 2.0 * dM_SiO2_er)) + M_FeSiO3 * (M_SiO2 * (
                      -2.0 * dM_FeSiO3_er - 2.0 * dM_MgSiO3_er - 2.0 * dM_SiO2_er) + M_m * (
                                                                                              -2.0 * dM_FeO_er - 2.0 * dM_MgO_er + 2.0 * dM_SiO2_er)))))) / (
               M_O * (M_Fe * (M_MgO * (4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                              M_FeO * (M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                              M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                              -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m))) + M_Mg * (M_Fe * (M_FeSiO3 * (
               M_FeO * (-4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (-4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO)) + M_MgO * (
                                                                                           4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + M_MgO * (
                                                                                           -4.0 * M_SiO2 + 4.0 * M_m) + 4.0 * M_SiO2 * M_m)))) + M_Si * (
               M_Fe * (
               M_MgO * (M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (M_MgO * (
               M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               -9.0 * M_MgO - M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                                                                                             -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                                                                                             -9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m))) + M_Mg * (
               M_Fe * (M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_O * (
                       M_FeO * (-M_SiO2 + 4.0 * M_m) + M_FeSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_MgO * (
                       -M_SiO2 + 4.0 * M_m) + M_MgSiO3 * (
                       -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_SiO2 * M_m * (
                       M_FeO + M_MgO)) + M_MgO * (
               M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-M_SiO2 + M_m) + M_MgO * (-M_SiO2 + M_m) + M_SiO2 * M_m)) + M_O * (
               M_FeO * (M_MgO * (-M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (-9.0 * M_MgO - 9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (
               -M_SiO2 + 4.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (-9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m) + M_FeSiO3 * (
               -9.0 * M_FeO - 9.0 * M_MgO - 4.0 * M_SiO2 + 25.0 * M_m)))) + M_O * (M_MgO * (
               9.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)) + M_MgSiO3 * (M_FeO * (
               M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m) + M_FeSiO3 * (M_FeO * (
               -9.0 * M_SiO2 + 9.0 * M_m) + M_MgO * (-9.0 * M_SiO2 + 9.0 * M_m) + 9.0 * M_SiO2 * M_m)))) + M_c * (
               M_Fe * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
               M_MgO * (M_FeSiO3 * (M_FeO - M_m) + M_SiO2 * (M_FeO - M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO + M_SiO2) + M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgO * (
               M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Mg * (M_Fe * (
               M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_MgSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_O * (
               M_FeSiO3 * (M_FeO + M_MgO - M_m) + M_MgSiO3 * (M_FeO + M_MgO - M_m) + M_SiO2 * (
               M_FeO + M_MgO - M_m)) + M_SiO2 * M_m * (-M_FeO - M_MgO)) + M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_SiO2 - M_m) + M_MgO * (
                                                         M_SiO2 - M_m) - M_SiO2 * M_m)) + M_O * (
                                                         M_FeO * M_SiO2 * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO * (M_MgO + M_SiO2 - M_m) + M_SiO2 * (
                                                         M_MgO - M_m)) + M_MgSiO3 * (
                                                         M_FeO * (M_MgO - M_m) + M_FeSiO3 * (
                                                         M_FeO + M_MgO - M_m)))) + M_O * (
               M_MgO * (-M_FeO * M_SiO2 * M_m + M_FeSiO3 * (M_FeO * (M_SiO2 - M_m) - M_SiO2 * M_m)) + M_MgSiO3 * (
               M_FeO * (M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m) + M_FeSiO3 * (
               M_FeO * (M_SiO2 - M_m) + M_MgO * (M_SiO2 - M_m) - M_SiO2 * M_m))) + M_Si * (M_Fe * (M_MgO * (
               M_FeO * (M_SiO2 - M_m) + M_FeSiO3 * (
               4.0 * M_FeO + M_SiO2 - 9.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (M_FeO * (
               4.0 * M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                     4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_O * (
                                                                                                   M_MgO * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   M_FeO + M_FeSiO3 + M_SiO2 - M_m))) + M_Mg * (
                                                                                           M_Fe * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_MgO * (
                                                                                                   M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                                   4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_O * (
                                                                                                   M_FeO + M_FeSiO3 + M_MgO + M_MgSiO3 + M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeO * (
                                                                                           M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + 4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           M_SiO2 - M_m) - 4.0 * M_SiO2 * M_m) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_MgO + M_SiO2 - 9.0 * M_m) + M_FeSiO3 * (
                                                                                           4.0 * M_FeO + 4.0 * M_MgO + M_SiO2 - 9.0 * M_m)) + M_O * (
                                                                                           M_FeO * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_MgO + M_SiO2 - M_m) + M_MgSiO3 * (
                                                                                           M_FeO + M_FeSiO3))) + M_MgO * (
                                                                                           -4.0 * M_FeO * M_SiO2 * M_m + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m) + M_FeSiO3 * (
                                                                                           M_FeO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) + M_MgO * (
                                                                                           4.0 * M_SiO2 - 4.0 * M_m) - 4.0 * M_SiO2 * M_m)) + M_O * (
                                                                                           M_MgO * (M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                                    M_SiO2 - M_m)) + M_MgSiO3 * (
                                                                                           M_FeO * (
                                                                                           M_SiO2 - M_m) + M_FeSiO3 * (
                                                                                           M_SiO2 - M_m))))))

    def dM_m_dTc_dMi(self, dM_im_dTc):
        '''alternate way to compute dM_m given the results of the mole changes of all mantle components
        inputs:
        dMi_m: [dM_MgO, dM_SiO2, dM_FeO, dM_MgSiO3, dM_FeSiO3]
        '''
        return np.sum(dM_im_dTc)

class MgO_Layer():
    def __init__(self, params = None):
        if params is None:
            params = Parameters('MgO layer parameters')
        self.params = params
        self.params.layer = Parameters('MgO layer parameters')
        pl = self.params.layer
        Cyr2s = 365.25*24*3600
        pl.C_b = 0.9 # [-] Mg/Fe molar ratio of background lower mantle
        pl.C_MgO = 0.91 # [-] Mg/Fe molar ratio of exsolved material
        pl.thickness = 300 # [m] thickness of layer
        pl.time_overturn = 800e6*Cyr2s # [s] overturn time of layer
        pl.V_c = 4/3*np.pi*3480e3**3 # [m^3] volume of total core
        pl.V_l = 4/3*np.pi*(3480e3+pl.thickness)**3 - pl.V_c # [m^3] volume of layer
        pl.V_r = pl.V_c / pl.V_l # [-] Ratio of volume of core to volume of layer
        pl.d = 1. # [-] exponent of layer overturn expression
        pl.e = pl.time_overturn / (1 + pl.d) # [s] constant of layer overturn expression
        pl.P = 135e9 # [Pa] pressure at CMB
        pl.time_solidify = 610e6*Cyr2s
        self.s = MgBadro()

    def dCdt_erode(self, C):
        pl = self.params.layer
        return np.sign(pl.C_b - C) * np.abs(pl.C_b - C) ** pl.d / pl.e

    def dCdt_exsolve(self, XMg, XO, C, T_cmb, dTcmbdt):
        pl = self.params.layer
        return (C- pl.C_MgO) * pl.V_r * self.dXMgdt(XMg, XO, C, T_cmb, dTcmbdt)

    def dCdt(self, t, XMg, XO, C, T_cmb, dTcmbdt):
        pl = self.params.layer
        dCdt = self.dCdt_erode(C) + self.dCdt_exsolve(XMg, XO, C, T_cmb, dTcmbdt)
        return (sp.erf((t-pl.time_solidify)/pl.time_solidify*100)/2+0.5)*dCdt
        # return dCdt

    def dXMgdt(self, XMg, XO, C, T_cmb, dTcmbdt):
        pl = self.params.layer
        P = 135e9
        Mg_sol = self.s.solubility(P, T_cmb, X_MgO=C, X_O=XO)
        T1 = self.s.b * np.log(10) / T_cmb ** 2
        C1 = np.sign(pl.C_b - C) * np.abs(pl.C_b - C) ** pl.d / pl.e
        C2 = (C - pl.C_MgO) * pl.V_r
        Mgex = (C1*XMg / C + T1 * dTcmbdt * XMg / (1 + XMg / XO - C2 * XMg / C))
        return (sp.erf((XMg - Mg_sol) / XMg * 100) / 2 + 0.5) * Mgex

    def dXOdt(self, XMg, XO, C, T_cmb, dTcmbdt):
        return self.dXMgdt(XMg, XO, C, T_cmb, dTcmbdt)
