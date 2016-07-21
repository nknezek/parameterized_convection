import numpy as np
import matplotlib.pyplot as plt


class IsotopeReservoir():
    def __init__(self, element, a, b, mol, isotope_ratio, amu_top, amu_bottom):
        self.element=element
        self.a = a
        self.b = b
        self.mol = mol
        self.isotope_ratio = isotope_ratio
        self.amu_top = amu_top
        self.amu_bottom = amu_bottom
        self.mol_layer = 0.
        self.isotope_ratio_layer = 1.

    def mass_to_mol(self, mass, isotope_ratio=None, amu_top=None, amu_bottom=None):
        if isotope_ratio is None:
            isotope_ratio = self.isotope_ratio
        if amu_top is None:
            amu_top = self.amu_top
        if amu_bottom is None:
            amu_bottom = self.amu_bottom
        return mass*((1 + isotope_ratio)/(amu_bottom + isotope_ratio*amu_top))

    def mol_to_mass(self, mol, isotope_ratio=None, amu_top=None, amu_bottom=None):
        if isotope_ratio is None:
            isotope_ratio = self.isotope_ratio
        if amu_top is None:
            amu_top = self.amu_top
        if amu_bottom is None:
            amu_bottom = self.amu_bottom
        return mol*(amu_bottom + isotope_ratio*amu_top)/(1+isotope_ratio)

    def get_mass(self):
        return self.mol_to_mass(self.mol, self.isotope_ratio, self.amu_top, self.amu_bottom)

    def get_mol(self):
        return self.mol

    def set_mol(self, mol):
        self.mol = mol

    def set_mass(self, mass):
        self.mol = self.mass_to_mol(mass)

    def set_isotope_ratio(self, isotope_ratio):
        self.isotope_ratio = isotope_ratio

    def get_isotope_ratio(self):
        return self.isotope_ratio

    def compute_isotope_ratio_in_product(self, fractionation_factor, f):
        isotope_ratio_prod = self.isotope_ratio*f**(fractionation_factor-1)
        return isotope_ratio_prod

    def compute_mol_fraction_out(self, mol_MgO):
        f = mol_MgO/self.get_mol()
        return f

    def compute_mol_of_isotopes(self, mol_total, isotope_ratio):
        mol_top = mol_total*isotope_ratio/(1+isotope_ratio)
        mol_bottom = mol_total/(1+isotope_ratio)
        return mol_top, mol_bottom

    def update_reservoir(self, mol_out, fractionation_factor):
        f = self.compute_mol_fraction_out(mol_out)
        isotope_ratio_out = self.compute_isotope_ratio_in_product(fractionation_factor, f)
        mol_out_top, mol_out_bottom = self.compute_mol_of_isotopes(mol_out, isotope_ratio_out)
        mol_current_top, mol_current_bottom = self.compute_mol_of_isotopes(self.get_mol(), self.isotope_ratio)
        mol_residue_top = mol_current_top - mol_out_top
        mol_residue_bottom = mol_current_bottom - mol_out_bottom
        isotope_ratio_residue = mol_residue_top/mol_residue_bottom
        mol_residue = mol_residue_bottom+mol_residue_top
        self.set_isotope_ratio(isotope_ratio_residue)
        self.set_mol(mol_residue)

        # update MgO layer
        mol_layer_top, mol_layer_bottom = self.compute_mol_of_isotopes(self.get_layer_mol(), self.get_layer_isotope_ratio())
        mol_layer_top_new = mol_layer_top + mol_out_top
        mol_layer_bottom_new = mol_layer_bottom + mol_out_bottom
        self.set_layer_mol(mol_layer_bottom_new + mol_layer_top_new)
        self.set_layer_isotope_ratio(mol_layer_top_new/mol_layer_bottom_new)

    def set_layer_mol(self, new_mol):
        self.mol_layer = new_mol

    def set_layer_isotope_ratio(self, isotope_ratio):
        self.isotope_ratio_layer = isotope_ratio

    def get_layer_mol(self):
        return self.mol_layer

    def get_layer_isotope_ratio(self):
        return self.isotope_ratio_layer

    def compute_fractionation_factor(self, T, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return np.exp(a + b*1e6/T**2)

    def compute_fractionation_factor2(self, T, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        return np.exp(a + b*1e6/T**2)

    def compute_time_history(self, times, temperatures, mol_out):
        alphas = np.zeros_like(times)
        layer_moles = np.zeros_like(times)
        layer_isotope = np.zeros_like(times)
        core_moles = np.zeros_like(times)
        core_isotope = np.zeros_like(times)
        for i,(t, T, mol) in enumerate(zip(times,temperatures,mol_out)):
            alpha = self.compute_fractionation_factor(T)
            self.update_reservoir(mol, alpha)
            alphas[i] = alpha
            layer_moles[i] = self.get_layer_mol()
            layer_isotope[i] = self.get_layer_isotope_ratio()
            core_moles[i] = self.get_mol()
            core_isotope[i] = self.get_isotope_ratio()
        return layer_moles, layer_isotope, core_moles, core_isotope, alphas

class MgReservoir(IsotopeReservoir):
    def __init__(self, a, b, mol, isotope_ratio):
        element = 'Mg'
        amu_top = 26
        amu_bottom = 24
        super(MgReservoir,self).__init__(element, a, b, mol, isotope_ratio, amu_top, amu_bottom)
