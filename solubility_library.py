"""
Created on Sun Jul 17 17:11:58 2016

@author: nknezek
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

class Solubility_calc():
    def __init__(self, element=None):
        self.element = element
        self.m_Mg = 24.305
        self.m_O = 15.9994
        self.m_Fe = 55.845
        self.m_MgO = self.m_Mg+self.m_O
        self.mass = None

    def solubility_to_wt(self, solubility):
        M_core = self.mass*solubility+ self.m_Fe*(1-solubility)
        wt = self.mass*solubility/M_core
        return wt

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

    def solubility(self,P,T):
        return self.solubility_OxyRatio(T)

    def solubility_OxyRatio(self, T, beta=1, X_MgO=0.5, a = None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return beta**(-0.5) * X_MgO**0.5 * np.power(10., (a - b/T)*0.5)

    def solubility_OxyConst(self, T, X_O, X_MgO=0.5, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return X_MgO/X_O * np.power(10., a - b/T)

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



