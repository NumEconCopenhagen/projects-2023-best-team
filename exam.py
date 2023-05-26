import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize
#Q1.1
def q11_solve_equation():
    # We write up our paramaters as symbols
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


#Q1.2.
# The initial paramters we were given are:
alpha = 0.5
nu = 1 / (2 * 16**2)
kappa = 1
tau = 0.3

# We write up the equilibrium L
def L_star(w, tau, kappa, alpha, nu):
    w_tilde = (1 - tau) * w
    return (-kappa + np.sqrt(kappa**2 + 4 * (alpha / nu) * w_tilde**2)) / (2 * w_tilde)

# create an array of w values
w_values = np.linspace(0.1, 2, 100)  # we start from 0.1 to avoid dividing by zero in L_star

# compute L* for each w
L_values = L_star(w_values, tau, kappa, alpha, nu)

# plot L* as a function of w
def q12_plot_optimal_labor():
    plt.figure(figsize=(10, 6))
    plt.plot(w_values, L_values, label='L*(w̃)')
    plt.xlabel('w')
    plt.ylabel('L*')
    plt.title('Optimal labor supply as a function of real wage')
    plt.legend()
    plt.grid(True)
    plt.show()

#q.1.3.

# Define the variables and parameters
w = 1.0
kappa = 1.0
alpha = 0.5
nu = 1 / (2 * 16**2)
tau_values = np.linspace(0, 1, 100)  # Grid of tau values

# Calculate implied L, G, and utility for each tau value
def q13_calculate_implications():
    # Initialize arrays to store results
    L_values = []
    G_values = []
    utility_values = []

    # Calculate implied L, G, and utility for each tau value
    for tau in tau_values:
        tilde_w = (1 - tau) * w
        L_star = (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * tilde_w**2)) / (2 * tilde_w)
        G = tau * w * L_star
        utility = np.log((kappa + (1 - tau) * w * L_star)**alpha * G**(1 - alpha)) - nu * L_star**2 / 2

        L_values.append(L_star)
        G_values.append(G)
        utility_values.append(utility)

    return L_values, G_values, utility_values

# Plot the results
def q13_plot_implications():
    L_values, G_values, utility_values = q13_calculate_implications()
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values, L_values, label='L')
    plt.plot(tau_values, G_values, label='G')
    plt.plot(tau_values, utility_values, label='Utility')
    plt.xlabel('Tax Rate (tau)')
    plt.ylabel('Value')
    plt.title('Implications of tau on L, G, and Utility')
    plt.legend()
    plt.grid(True)
    plt.show()
#q1.4.

# Define the variables and parameters
w = 1.0
kappa = 1.0
alpha = 0.5
nu = 1 / (2 * 16**2)

# Define the worker utility function
def q14_worker_utility(tau):
    tilde_w = (1 - tau) * w
    L_star = (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * tilde_w**2)) / (2 * tilde_w)
    utility = np.log((kappa + (1 - tau) * w * L_star)**alpha * (tau * w * L_star)**(1 - alpha)) - nu * L_star**2 / 2
    return -utility  # Negative sign for maximization

def q14_plot_graph():
    # Plot the worker utility function
    tau_values = np.linspace(0, 1, 100)

    # Get the optimal tax rate and utility
    tau_star, utility_star = q14_find_optimal_tax()

    # Plot the worker utility function
    utility_values = [-q14_worker_utility(tau) for tau in tau_values]

    plt.figure(figsize=(8, 6))
    plt.plot(tau_values, utility_values)
    plt.scatter(tau_star, utility_star, color='red', label='Maximized Utility')
    plt.xlabel('Tax Rate (tau)')
    plt.ylabel('Worker Utility')
    plt.title('Worker Utility as a Function of Tax Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

def q14_find_optimal_tax():
    # Perform numerical optimization to find the optimal tau
    result = minimize(q14_worker_utility, x0=0.5, bounds=[(0, 1)])

    tau_star = result.x[0]
    utility_star = -result.fun

    return tau_star, utility_star

if __name__ == "__main__":
    tau_star, utility_star = q14_find_optimal_tax()
    print("Optimal tax rate (tau_star):", tau_star)
    print("Maximized worker utility:", utility_star)
    q14_plot_graph()


#q.1.5. & q.1.6.
import numpy as np
from scipy.optimize import minimize, fsolve

# Define the variables and parameters
alpha = 0.5
kappa = 1.0
nu = 1 / (2 * 16**2)
w = 1.0

# Define parameter sets
set1 = {'sigma': 1.001, 'rho': 1.001, 'epsilon': 1.0}
set2 = {'sigma': 1.5, 'rho': 1.5, 'epsilon': 1.0}

# Define utility function
def utility(L, G, tau, params):
    C = kappa + (1 - tau) * w * L
    term1 = (alpha * C ** ((params['sigma'] - 1) / params['sigma']) + (1 - alpha) * G ** (
                (params['sigma'] - 1) / params['sigma'])) ** (params['sigma'] / (params['sigma'] - 1))
    term2 = nu * L ** (1 + params['epsilon']) / (1 + params['epsilon'])
    return (term1 ** (1 - params['rho']) - 1) / (1 - params['rho']) - term2

# Find optimal L for given G and tau
def find_L_star(G, tau, params):
    result = minimize(lambda L: -utility(L, G, tau, params), [12.0], method='L-BFGS-B', bounds=[(0, 24)])
    return result.x[0]

# Find G that solves the given equation
def find_optimal_G(tau, params):
    return fsolve(lambda G: G - tau * w * find_L_star(G, tau, params), [1])

# Find socially optimal tax rate
def find_optimal_tau(params):
    result = minimize(lambda tau: -social_optimum(tau, params), [0.51], method='L-BFGS-B', bounds=[(0, 1)])
    return result.x[0]

# Social optimum function
def social_optimum(tau, params):
    G = find_optimal_G(tau, params)
    return -utility(find_L_star(G, tau, params), G, tau, params)

def get_optimal_G_and_tau(params):
    # Question 5: Find G that solves the given equation
    tau_q5 = 0.54
    G_q5 = find_optimal_G(tau_q5, params)

    # Question 6: Find socially optimal tax rate
    tau_star = find_optimal_tau(params)

    return G_q5, tau_star

if __name__ == "__main__":
    G_q5_set1, tau_star_set1 = get_optimal_G_and_tau(set1)
    G_q5_set2, tau_star_set2 = get_optimal_G_and_tau(set2)

    print(f"Optimal G for Set 1: {G_q5_set1[0]}, Optimal Tax Rate for Set 1: {tau_star_set1}")
    print(f"Optimal G for Set 2: {G_q5_set2[0]}, Optimal Tax Rate for Set 2: {tau_star_set2}")

#q.2.1.

#q.2.2.

#q.2.3.

#q.2.4.
# Define the baseline parameters
import numpy as np
import matplotlib.pyplot as plt

# Define the baseline parameters
rho = 0.90
iota = 0.01
sigma_epsilon = 0.10
R = (1 + 0.01)**(1 / 12)
eta = 0.5
w = 1.0
T = 120
K = 10000

def q24_calculate_H_values(Delta_values):
    # Initialize H_values array
    H_d5_values = np.zeros(len(Delta_values))

    # Generate K shock series
    np.random.seed(0)  # for reproducibility
    shock_series = np.random.normal(loc=-0.5 * sigma_epsilon**2, scale=sigma_epsilon, size=(K, T))

    # Loop over Delta values
    for i, Delta in enumerate(Delta_values):
        h_values = np.zeros(K)
        for k in range(K):
            # Initialize kappa and l
            kappa = np.zeros(T)
            kappa[0] = np.exp(rho * np.log(1) + shock_series[k, 0])  # initial kappa
            l = np.zeros(T)
            l_star_initial = ((1 - eta) * kappa[0] / w)**(1 / eta)
            l[0] = l_star_initial  # initial l

            # Calculate l_t and kappa_t for t = 1, ..., T-1
            for t in range(1, T):
                kappa[t] = np.exp(rho * np.log(kappa[t - 1]) + shock_series[k, t])
                l_star = ((1 - eta) * kappa[t] / w)**(1 / eta)
                # Update l[t] based on the policy
                if np.abs(l[t - 1] - l_star) > Delta:
                    l[t] = l_star
                else:
                    l[t] = l[t - 1]

            # Calculate h for this shock series
            adjustment_cost = iota * np.sum(l[1:] != l[:-1])
            profit = np.sum((kappa * l**(1 - eta) - w * l) * R**-np.arange(T)) - adjustment_cost
            h_values[k] = profit

        # The ex-ante expected value H is the average of these values
        H_d5_values[i] = np.mean(h_values)

    return H_d5_values

def q24_plot_H_vs_Delta(Delta_values, H_values):
    # Plot H against Delta
    plt.figure(figsize=(10, 6))
    plt.plot(Delta_values, H_values, label='H')
    plt.xlabel(r'$\Delta$')
    plt.ylabel('H')
    plt.title('H vs Delta')
    plt.legend()
    plt.grid(True)
    plt.show()

def q24_find_optimal_Delta(Delta_values, H_values):
    # Find the Delta value that maximizes H
    optimal_Delta = Delta_values[np.argmax(H_values)]
    return optimal_Delta



#q.2.5.

#q.3.1.


#q.3.2.
