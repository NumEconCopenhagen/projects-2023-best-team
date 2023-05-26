#import libraries, modules, and classes
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self, epsilon_M=None, epsilon_F=None, nu_M=None, nu_F=None):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace() # Creating a namespace for parameters
        sol = self.sol = SimpleNamespace() # Creating a namespace for solution variables

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.nu_M = 0.001 if nu_M is None else nu_M # Set the parameters with default values if not provided
        par.nu_F = 0.001 if nu_F is None else nu_F
        par.epsilon = 1.0
        par.epsilon_M = 1.0 if epsilon_M is None else epsilon_M
        par.epsilon_F = 1.0 if epsilon_F is None else epsilon_F
        par.omega = 0.5
        print(par.nu_M, par.nu_F, par.epsilon_M, par.epsilon_F)

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0  # Set the wage for males
        par.wF = 1.0  # Set the wage for females
        par.wF_vec = np.linspace(0.8,1.2,5) # Create an array of female wages from 0.8 to 1.2

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size) # Initialize an array of zeros for LM
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan # Initialize beta0 as NaN (not a number)
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par # Access the parameters namespace
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if np.allclose(par.sigma, 0): # if par.sigma==0: can't compare float with equality
            H=np.min([HM,HF]) # Set H as the minimum of HM and HF
        elif np.allclose(par.sigma, 1): # elif par.sigma==1: can't compare float with equality
            H = HM**(1-par.alpha) * HF**par.alpha # Calculate H based on the formula for sigma = 1
        else:
            H = ((1 - par.alpha) * HM**((par.sigma - 1) / par.sigma) + par.alpha * HF**((par.sigma - 1) / par.sigma))**(par.sigma / (par.sigma - 1))

        
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho) # Calculate the utility

        # d. disutlity of work
        epsilon_M = 1 + 1 / par.epsilon_M  # Calculate epsilon_M
        epsilon_F = 1 + 1 / par.epsilon_F  # Calculate epsilon_F
        TM = LM + HM  # Calculate the total work time for males
        TF = LF + HF  # Calculate the total work time for females
        disutility = par.nu_M * (TM**epsilon_M / epsilon_M) + par.nu_F * (TF**epsilon_F / epsilon_F)  # Calculate the disutility of work
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par # Access the parameters namespace
        sol = self.sol
        opt = SimpleNamespace()  # Create a namespace for the optimal solution
        
        # a. all possible choices
        x = np.linspace(0,24,49)  # Create an array of values from 0 to 24
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # create a meshgride all combinations
    
        LM = LM.ravel() # Flatten LM into a vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)  # Calculate the utility for each combination of LM, HM, LF, and HF
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or" # Check if the constraint is broken for any combination
        u[I] = -np.inf # if yes, set the utility to minus infinity
    
        # d. find maximizing argument
        j = np.argmax(u) # Find the index of the combination that maximizes the utility
        
        opt.LM = LM[j]  # Set the optimal value for LM
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}') # Print the optimal values

        return opt # Return the optimal solution
    

 #Continuous:   
    def solve(self):
              
        def obj(x):
            u = self.calc_utility(x[0], x[1], x[2], x[3]) # Calculate the utility based on the given values
            return - u # Return the negative utility (to be minimized)
    
        bounds = [(0, 24)]*4  # Define the bounds for LM, HM, LF, and HF as (0, 24)
        guess = [6.0]*4 # Initial guess for LM, HM, LF, and HF
# call the numerical minimizer
        solution = optimize.minimize(obj, x0 = guess, bounds=bounds) #options={'xatol': 1e-4}) # Find the numerical solution that minimizes the objective function

       
        return solution.x

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par # Access the parameters namespace
        sol = self.sol 
        WF_list = [0.8, 0.9, 1., 1.1, 1.2]
        for it, alpha in enumerate(WF_list):
            par.wF = alpha # Set the current female wage
            out = self.solve() # Solve the model
            sol.LM_vec[it] = out[0] # Store the optimal value of LM
            sol.HM_vec[it] = out[1]      
            sol.LF_vec[it] = out[2]
            sol.HF_vec[it] = out[3]      
        
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec) # Take the logarithm of the vector of female wages
        y = np.log(sol.HF_vec/sol.HM_vec) # Calculate the logarithm of the ratio of HF to HM
        A = np.vstack([np.ones(x.size),x]).T # Create a matrix A with columns of ones and x values
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0] # Perform least squares regression to estimate beta0 and beta1

    

#estimate function to allow for different values of epislon and nu:
  
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        def obj(x):
            par = self.par #access namespaces
            sol = self.sol
            par.nu_M = x[0] #set parameters
            par.nu_F = x[1]            
            par.sigma = x[2]
            par.epsilon_M = x[3]
            par.epsilon_F = x[4]            
            self.solve_wF_vec() # Solve the model for a vector of female wages
            self.run_regression() # Run regression to estimate beta0 and beta1
            fun = (par.beta0_target - sol.beta0)**2.0 + (par.beta1_target - sol.beta1)**2.0 # Calculate the objective function value

            return fun

        bounds = [(0., 24.), (0., 24.),  (0., 24.),  (0., 24.), (0., 24.)] #[(min_alpha, max_alpha), (min_sigma, max_sigma)]
        guess = [.001, .001, 1, 1, 1] #[alpha, sigma] # Initial guess for nu_M, nu_F, sigma, epsilon_M, and epsilon_F
        solution = optimize.minimize(obj, x0 = guess, bounds=bounds, method = "nelder-mead") #options={'xatol': 1e-4}) # Find the numerical solution that minimizes the objective function


        
        return solution.x  # Return the optimal values for nu_M, nu_F, sigma, epsilon_M, and epsilon_F
    
    def estimate1(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        
        def obj(x):
            par = self.par  # Assign the value of self.par to the variable par
            sol = self.sol  # Assign the value of self.sol to the variable sol
            par.alpha = x[0]  # Set the value of par.alpha to the first element of the x list
            par.sigma = x[1]  # Set the value of par.sigma to the second element of the x list
            self.solve_wF_vec()  # Invoke the solve_wF_vec() method on the current instance
            self.run_regression()  # Invoke the run_regression() method on the current instance
            fun = (par.beta0_target - sol.beta0)**2.0 + (par.beta1_target - sol.beta1)**2.0  # Calculate the value of fun
            
            return fun

        bounds = [(0., 24.), (0., 24.)]  # Define a list of bounds [(min_alpha, max_alpha), (min_sigma, max_sigma)]
        guess = [.5, 1]  # Define a list of initial guesses [alpha, sigma]
        solution = optimize.minimize(obj, x0=guess, bounds=bounds, method="nelder-mead")  # Find the minimum of the obj function using the given bounds and initial guesses
            # using the Nelder-Mead optimization method
            # options={'xatol': 1e-4}) can be used to specify additional options if needed

        return solution.x  # Return the solution found by the optimization algorithm
    
        pass # End of the class definition

   
