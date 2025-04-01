import pandas as pd
import numpy as np 
from math import sin, cos, sqrt
from scipy.constants import hbar, physical_constants as const
import tkinter as tk
from tkinter import simpledialog, messagebox
import matplotlib.pyplot as plt
import sys

# number of default samples to plot 
default_samples = int(1e6)

def ev_to_j(e_ev: float) -> float:
    j_per_ev = const['electron volt-joule relationship'][0]
    return e_ev * j_per_ev

def j_to_ev(e_j: float) -> float:
    ev_per_j = const['joule-electron volt relationship'][0]
    return e_j * ev_per_j

def b_const(l: float, V_0: float) -> float:
    m = const['electron mass'][0]
    return sqrt(2 * m * V_0) * l / hbar  


def func_bound_state(b: float, e: float, cos_e: float, sin_e: float, 
                        sqrt_e_min_e_sqr: float) -> float:
    return (2*e - 1) * sin_e - 2 * sqrt_e_min_e_sqr * cos_e

def func_prime_bound_state(b: float, e: float, cos_e: float, sin_e: float, 
                            sqrt_e_min_e_sqr: float, sqrt_e: float) -> float:
    term_1 = 2 * sin_e  
    term_2 = (2*e-1) * cos_e*(b/(2*sqrt(e)))
    term_3 = - (1-2*e)  * cos_e / (sqrt_e_min_e_sqr) 
    term_4 = b * sin_e * sqrt_e_min_e_sqr / sqrt_e
    return term_1 + term_2 + term_3 + term_4

def calc_constants(e: float, length_m: float, V0_J: float) -> tuple:
    sqrt_e = sqrt(e)
    b = b_const(length_m, V0_J)
    sin_e = sin(b * sqrt_e)
    cos_e = cos(b * sqrt_e)
    sqrt_e_min_e_sqr = sqrt(e - e**2)
    return b, sin_e, cos_e, sqrt_e, sqrt_e_min_e_sqr

def data(V0_J: float, length_m: float, samples: int) -> tuple[pd.DataFrame, list]:
    data_dict = {}
    data_dict['epsilons'] = np.linspace(start=0, stop=1, num=samples, endpoint=False)
    data_dict['values'] = []
    allowed = []
    
    tolerance = 1e-10
    
    for idx, e in enumerate(data_dict['epsilons']):
        
        b, sin_e, cos_e, sqrt_e, sqrt_e_min_e_sqr = calc_constants(e, length_m, V0_J)
        val = func_bound_state(b, e, cos_e, sin_e, sqrt_e_min_e_sqr)
        data_dict['values'].append(val)
            
        if abs(val) < tolerance:
            # E <= 0 are invalid solutions 
            if e != 0:
                allowed.append(e)
        elif idx > 0:
            val_prev = data_dict['values'][idx - 1]
            e_prev = data_dict['epsilons'][idx - 1]
            crossed_zero = (val < 0 and val_prev > 0) or (val > 0 and val_prev < 0)
            if crossed_zero:
                r = (e + e_prev) / 2
                while True: 
                    b_r, sin_r, cos_r, sqrt_r, sqrt_r_min_r_sqr = calc_constants(r, length_m, V0_J)
                    bound = func_bound_state(b_r, r, cos_r, sin_r, sqrt_r_min_r_sqr)
                    bound_prime = func_prime_bound_state(b_r, r, cos_r, sin_r, 
                                sqrt_r_min_r_sqr, sqrt_r)
                    if abs(bound_prime) < 1e-14:
                        break  
                    new_r = r - bound / bound_prime
                    if new_r <= 0 or new_r >= 1:
                        break
                    if abs(new_r - r) < tolerance:
                        r = new_r
                        break
                    r = new_r
                allowed.append(r)
    return pd.DataFrame.from_dict(data_dict), allowed

def calc_energy_ev(V0_J: float, allowed_epsilons: list) -> list:
    return [j_to_ev(e * V0_J) for e in allowed_epsilons]

def error_prompt(msg: str, condition: bool, msg_title=None) -> int:
    if not condition:
        messagebox.showerror(msg_title, message=msg)
        return 1
    return 0

def quit(event):
    root.destroy()

if __name__ == "__main__":
    program_title = "Finite Potential Well Solver for an H Atom"

    root = tk.Tk()
    root.bind('<Escape>', quit)
    root.withdraw()
    
    # prompt for potential barrier (V0)
    pmt = 'Please enter the potential barrier (V0) in eV:'
    err = 'Invalid potential! Must be a positive number.'
    V0_eV = simpledialog.askfloat(title=program_title, prompt=pmt)
    if error_prompt(err, isinstance(V0_eV, (float, int)) and V0_eV > 0, 
                    msg_title=program_title):
        root.update()
        root.mainloop() 

    # prompt for barrier width (length_m) 
    pmt = 'Please enter the barrier width (nm):'
    err = 'Invalid barrier width! Must be a positive number.'
    length_nm = simpledialog.askfloat(title=program_title, prompt=pmt)
    if error_prompt(err, isinstance(length_nm, (float, int)) and length_nm > 0, 
                    msg_title=program_title):
        root.update()
        root.mainloop() 
    length_m = length_nm * 1e-9 

    # number of samples
    samples = default_samples
    pmt = f'Keep default sample numbers? (Default is {default_samples})'
    if not messagebox.askyesno(title=program_title, message=pmt):
        pmt = 'Please enter desired number of samples:'
        err = 'Invalid samples! Must be a positive integer.'
        new_samples = simpledialog.askinteger(title=program_title, prompt=pmt)
        if error_prompt(err, isinstance(new_samples, int) and new_samples > 0, 
                        msg_title=program_title):
            root.update()
            root.mainloop()
        samples = new_samples
    
    # convert V0_eV -> V0_J
    V0_J = ev_to_j(V0_eV)

    # calculate
    df, allowed_epsilons = data(V0_J, length_m, samples)
    allowed_energies = calc_energy_ev(V0_J, allowed_epsilons)

    # notify calculation is complete
    messagebox.showinfo(program_title, "Calculation complete!")

    pmt = f'Enter name of save file (WARNING: WILL OVERWRITE FILES): '
    f_name = simpledialog.askstring(title=program_title, prompt=pmt)
    if f_name:
        # write allowed energies
        with open(f'{f_name}.txt', 'w') as f:
            f.write(f'Bound State: 0<E<V_0 (0<\u03B5<1)\n')
            f.write('E = \u03B5*V_0\n\n')
            f.write(f'Barrier width = L = {length_nm}(nm)\n')
            f.write(f'E_allowed when:\n') 
            l1 = f'(2*\u03B5-1)*sin((2*m_e*V_0)^(1/2)*(l/h_bar)*(\u03B5)^(1/2))'
            l2 = f'-(2*(\u03B5-\u03B5^2)^(1/2))*cos((2*m_e*V_0)^(1/2)*(l/h_bar)*(\u03B5)^(1/2))'
            f.write(f'\t{l1}{l2}=0\n\n')
            f.write(f'E_allowed_h_atom_eV\n') 
            for energy in allowed_energies:
                f.write(f'{energy}\n')
    # print allowed energy levels
    print("\nAllowed energies (eV):\n", allowed_energies)
    print("\n\n")
    
    
    if messagebox.askyesno(title=program_title, message='Show plot?'):
        fig, ax = plt.subplots(figsize=(10, 6))    
        ax.plot(df['epsilons'], df['values'], color='blue', linewidth=1)
        ax.set_title(r'Approximate Allowed Energies shown in red', fontsize=18)
        ax.set_xlabel('')
        ax.set_ylabel('')
        epsilon_max = df['epsilons'].max()
        values_min = df['values'].min()
        values_max = df['values'].max()
        y_padding = 0.1 * (values_max - values_min)
        ax.set_xlim(left=0, right=epsilon_max)
        ax.set_ylim(values_min - y_padding, values_max + y_padding)
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # plot approximated roots using allowed_epsilons on x-axis
        x_vals = allowed_epsilons
        y_vals = np.zeros(len(x_vals))
        ax.scatter(x_vals, y_vals, color='red')

        # annotate each point with its corresponding energy label (in eV)
        for x, energy in zip(x_vals, allowed_energies):
            ax.text(
                x, 0, f'{energy:.2f}eV', 
                ha='center', 
                va='top', 
                bbox=dict(
                    facecolor='lightcoral', 
                    alpha=0.8, 
                    edgecolor='Firebrick', 
                    boxstyle='round,pad=0.01'
                ),  
                fontsize=12, 
                rotation=-45
            )
        
        # caption
        l_0 = r'$0 < E < V_{0}$'
        l_1 =  r'$\quad\epsilon = \frac{E}{V_0}$'
        l_2 = rf'$\quadV_0 = {V0_eV}eV$' 
        l_3 = r'$\quadE_{allowed}$' + rf'$=\epsilon\times{V0_eV}eV$'
        l_4 = rf'$\quadL = {length_nm}nm$'
        label = l_0 + '\t' + l_1 + '\t' + l_2 + '\t' + l_3 + '\t' + l_4
        plt.figtext(
            0.5, 0.03, label, 
            ha='center', 
            va='bottom', 
            fontsize=18, 
            bbox=dict(
                facecolor='aliceblue', 
                alpha=0.5, 
                edgecolor='blue', 
                boxstyle='round,pad=0.5'
            )
        )
        plt.tight_layout()
        plt.show()
    sys.exit()
