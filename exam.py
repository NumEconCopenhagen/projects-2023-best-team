import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize

def q1_solve_equation():
    # Define the symbols
    C, L, w, tau, alpha, nu, kappa, G = sp.symbols('C L w tau alpha nu kappa G')

    # Define tilde{w}
    tilde_w = (1 - tau) * w

    # Define the utility function
    V = sp.log(C**alpha * G**(1 - alpha)) - nu*L**2 / 2

    # Substitute the constraint into the utility function
    V_sub = V.subs(C, kappa + tilde_w * L)

    # Take the derivative of the utility function
    V_prime = sp.diff(V_sub, L)

    # Solve for L where the derivative equals zero
    solution = sp.solve(V_prime, L)
    
    return solution

import numpy as np
import matplotlib.pyplot as plt

# define the parameter values
alpha = 0.5
nu = 1 / (2 * 16**2)
kappa = 1
tau = 0.3

# define a function for optimal labor supply
def L_star(w, tau, kappa, alpha, nu):
    w_tilde = (1 - tau) * w
    return (-kappa + np.sqrt(kappa**2 + 4 * (alpha / nu) * w_tilde**2)) / (2 * w_tilde)

# create an array of w values
w_values = np.linspace(0.1, 2, 100)  # we start from 0.1 to avoid dividing by zero in L_star

# compute L* for each w
L_values = L_star(w_values, tau, kappa, alpha, nu)

# plot L* as a function of w
def plot_optimal_labor():
    plt.figure(figsize=(10, 6))
    plt.plot(w_values, L_values, label='L*(w̃)')
    plt.xlabel('w')
    plt.ylabel('L*')
    plt.title('Optimal labor supply as a function of real wage')
    plt.legend()
    plt.grid(True)
    plt.show()

