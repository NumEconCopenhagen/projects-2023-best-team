import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize
#Q1.1
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


#Q1.2.
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

#q.1.3.
import numpy as np
import matplotlib.pyplot as plt

# Define the variables and parameters
w = 1.0
kappa = 1.0
alpha = 0.5
nu = 1 / (2 * 16**2)
tau_values = np.linspace(0, 1, 100)  # Grid of tau values

# Calculate implied L, G, and utility for each tau value
def calculate_implications():
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
def plot_implications():
    L_values, G_values, utility_values = calculate_implications()
    
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
def worker_utility(tau):
    tilde_w = (1 - tau) * w
    L_star = (-kappa + np.sqrt(kappa**2 + 4 * alpha / nu * tilde_w**2)) / (2 * tilde_w)
    utility = np.log((kappa + (1 - tau) * w * L_star)**alpha * (tau * w * L_star)**(1 - alpha)) - nu * L_star**2 / 2
    return -utility  # Negative sign for maximization

# Perform numerical optimization to find the optimal tau
result = minimize(worker_utility, x0=0.5, bounds=[(0, 1)])

tau_star = result.x[0]
utility_star = -result.fun

print("Optimal tax rate (tau_star):", tau_star)
print("Maximized worker utility:", utility_star)

# Plot the worker utility function
tau_values = np.linspace(0, 1, 100)
utility_values = [-worker_utility(tau) for tau in tau_values]

plt.figure(figsize=(8, 6))
plt.plot(tau_values, utility_values)
plt.scatter(tau_star, utility_star, color='red', label='Maximized Utility')
plt.xlabel('Tax Rate (tau)')
plt.ylabel('Worker Utility')
plt.title('Worker Utility as a Function of Tax Rate')
plt.legend()
plt.grid(True)
plt.show()


#q.1.5.

#q.1.6.

#q.2.1.
#q.2.2.
# Define the baseline parameters
rho = 0.90
iota = 0.01
sigma_epsilon = 0.10
R = (1 + 0.01)**(1 / 12)
eta = 0.5
w = 1
T = 120
K = 10000

def calculate_h():
    # Initialize the array to store the values of h for each shock series
    h_values = np.zeros(K)

    # Generate K shock series
    np.random.seed(0)  # for reproducibility
    shock_series = np.random.normal(loc=-0.5 * sigma_epsilon**2, scale=sigma_epsilon, size=(K, T))

    # Loop over the shock series
    for k in range(K):
        # Initialize kappa and l
        kappa = np.zeros(T)
        kappa[0] = np.exp(rho * np.log(1) + shock_series[k, 0])  # initial kappa
        l = np.zeros(T)
        l[0] = ((1 - eta) * kappa[0] / w)**(1 / eta)  # initial l

        # Calculate l_t and kappa_t for t = 1, ..., T-1
        for t in range(1, T):
            kappa[t] = np.exp(rho * np.log(kappa[t - 1]) + shock_series[k, t])
            l[t] = ((1 - eta) * kappa[t] / w)**(1 / eta)

        # Calculate h for this shock series
        adjustment_cost = iota * np.sum(l[1:] != l[:-1])
        profit = np.sum((kappa * l**(1 - eta) - w * l) * R**-np.arange(T)) - adjustment_cost
        h_values[k] = profit

    # The ex-ante expected value H is the average of these values
    H = np.mean(h_values)
    return H

#q.2.3.

# Define the baseline parameters
rho = 0.90
iota = 0.01
sigma_epsilon = 0.10
R = (1 + 0.01)**(1 / 12)
eta = 0.5
w = 1.0
T = 120
K = 10000

def calculate_h_new_policy():
    # Initialize the array to store the values of h for each shock series
    h_values = np.zeros(K)

    # Generate K shock series
    np.random.seed(0)  # for reproducibility
    shock_series = np.random.normal(loc=-0.5 * sigma_epsilon**2, scale=sigma_epsilon, size=(K, T))

    # Loop over the shock series
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
            if np.abs(l[t - 1] - l_star) > 0.05:
                l[t] = l_star
            else:
                l[t] = l[t - 1]

        # Calculate h for this shock series
        adjustment_cost = iota * np.sum(l[1:] != l[:-1])
        profit = np.sum((kappa * l**(1 - eta) - w * l) * R**-np.arange(T)) - adjustment_cost
        h_values[k] = profit

    # The ex-ante expected value H_new_policy is the average of these values
    H_new_policy = np.mean(h_values)
    return H_new_policy

def compare_policies(H):
    # Calculate H_new_policy
    H_new_policy = calculate_h_new_policy()

    # Compare with the previous policy
    profitability_improvement = H_new_policy - H

    return H_new_policy, profitability_improvement


#q.2.4.
#q.2.5.

#q.3.1.

#q.3.2.
