import numpy as np
import matplotlib.pyplot as plt
import input_parameters as params
import solubility_library as sol
import plotting_library as peplot
import planetary_energetics as pe
import isotope_library as iso
import dynamo_library as dyn
const_yr_to_sec = 31557600

p = params.Nimmo_2015()

## Calculate Solubility and Exsolution
P_cmb = 135e6 # Pa
# initial_concentration = 0.0095
# initial_concentration = 0.0115
X_Mg_0 = p.X_Mg_0
Tin = np.linspace(6500,3500,1000)
Pin = P_cmb*np.ones(len(Tin))

# Mg_sol = sol.MgDubrovinskaia()
# deg_fit = 1
# solubility = Mg_sol.solubility(Pin, Tin, deg=deg_fit)
Mg_sol = sol.MgBadro()
wt05b1 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.solubility(P_cmb, Tin, X_MgO=0.5, beta=1))
wt05O06 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.solubility(P_cmb, Tin, X_MgO=0.5, X_O=0.06))
wt05O10 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.solubility(P_cmb, Tin, X_MgO=0.5, X_O=0.1))
wt09O10 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.solubility(P_cmb, Tin, X_MgO=0.9, X_O=0.1))
wt02O10 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.solubility(P_cmb, Tin, X_MgO=0.2, X_O=0.1))
wt09O06 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.solubility(P_cmb, Tin, X_MgO=0.9, X_O=0.06))
wt02O06 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.solubility(P_cmb, Tin, X_MgO=0.2, X_O=0.06))

fig = plt.figure(figsize=(8,6))
plt.plot(Tin, wt05b1*100, 'k-', label='X_MgO=0.5, b=1')
plt.plot(Tin, wt09O06*100, 'b-', label='X_MgO=0.9, X_O=0.06')
plt.plot(Tin, wt05O06*100, 'r-', label='X_MgO=0.5, X_O=0.06')
plt.plot(Tin, wt02O06*100, 'g-', label='X_MgO=0.2, X_O=0.06')
plt.plot(Tin, wt09O10*100, 'b--', label='X_MgO=0.9, X_O=0.1')
plt.plot(Tin, wt05O10*100, 'r--', label='X_MgO=0.5, X_O=0.1')
plt.plot(Tin, wt02O10*100, 'g--', label='X_MgO=0.2, X_O=0.1')
plt.xlabel("Temperature (K)")
plt.xlim(3500, 5500)
plt.ylabel("MgO Solubility (wt%)")
plt.grid()
plt.ylim(0,6)
plt.title("MgO Solubility and Exsolution")
plt.legend(loc=0)
plt.savefig('Mg_solubility.pdf')

exBadro = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.5, beta=1))
exBadro09 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.9, beta=1))
ex090_006 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.9, X_O=0.06))
ex090_010 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.9, X_O=0.1))
ex020_006 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.2, X_O=0.06))
ex020_010 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.2, X_O=0.1))
ex050_006 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.5, X_O=0.06))
ex050_010 = Mg_sol.Mg_mol_frac_to_MgO_wtp(Mg_sol.exsolution(P_cmb, Tin, X_MgO=0.5, X_O=0.1))

C = 1e5

Orourke_nominal = 4e-5
Orourke_min = 2.5e-5
fig = plt.figure(figsize=(8,6))
plt.plot(Tin, np.ones_like(Tin)*Orourke_min*C, 'k-.', label='Orourke min')
plt.plot(Tin, np.ones_like(Tin)*Orourke_nominal*C, 'k--', label='Orourke Nominal')
plt.plot(Tin, exBadro*C, 'k', label='Badro')
plt.plot(Tin, ex090_006*C, 'b-', label='X_MgO=0.9, X_O=0.06')
plt.plot(Tin, ex090_010*C, 'b--', label='X_MgO=0.9, X_O=0.1')
plt.plot(Tin, ex050_006*C, 'r-', label='X_MgO=0.5, X_O=0.06')
plt.plot(Tin, ex050_010*C, 'r--', label='X_MgO=0.5, X_O=0.1')
plt.plot(Tin, ex020_006*C, 'g-', label='X_MgO=0.2, X_O=0.06')
plt.plot(Tin, ex020_010*C, 'g--', label='X_MgO=0.2, X_O=0.1')
plt.xlim(3500,5500)
plt.xlabel("Temperature (K)")
plt.ylabel("C_m or MgO Exsolution (1e-5 wt%/1K)")
plt.ylim(0,10)
plt.yticks(range(11))
plt.grid()
plt.title("C_m or wt% Exsolution from Core per 1K")
plt.legend(loc=0)
plt.savefig('Mg_exsolution.pdf')
