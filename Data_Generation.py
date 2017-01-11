import numpy as np
from scipy.stats import truncnorm
import math

def theta_generation(number_of_daytypes):
    """theta_generation : this function generates a theta according to the laws described in the article"""
    sigma2_s_param = 1 / (np.random.gamma(size = 1, shape = 0.01, scale = 100)[0]) 
    sigma2_g_param = 1 / (np.random.gamma(size = 1, shape = 0.01, scale = 100)[0]) 
    u_heat = np.random.normal(size=1, loc = 14, scale = 1)[0] #loc = mean / scale = standard deviation /!\
    kappa = (np.random.dirichlet(size = 1, alpha = [1]*number_of_daytypes)[0])*number_of_daytypes
    sigma2 = 1 / (np.random.gamma(size = 1, shape = 0.01, scale = 100)[0]) 
    return(sigma2_s_param,sigma2_g_param,u_heat,kappa,sigma2)

def x_season_generation(n,electricity_data,kappa, sigma_s_param, len_initialization = 1000):
    """x_season_generation : simulation of the x_season according to the law described in the article"""
    #Initialization
    x = np.zeros(n)
    nu = truncnorm.rvs(a = 0, b = math.inf, loc= 0, scale = sigma_s_param, size=1)[0]
    sigma_s_current = nu
    error = truncnorm.rvs(a = 0, b = math.inf, loc= 0, scale = sigma_s_current, size=1)[0]
    s_current=error
    for i in range(1,len_initialization,1):
        nu = truncnorm.rvs(a = -sigma_s_current / sigma_s_param , b = math.inf, loc= 0, scale = sigma_s_param, size=1)[0]
        sigma_s_current += nu
        error = truncnorm.rvs(a = -s_current / sigma_s_current , b = math.inf, loc= 0, scale = sigma_s_current, size=1)[0]
        s_current += error
    for i in range(0,n,1):
        nu = truncnorm.rvs(a = -sigma_s_current / sigma_s_param , b = math.inf, loc= 0, scale = sigma_s_param, size=1)[0]
        sigma_s_current += nu
        error = truncnorm.rvs(a = -s_current / sigma_s_current , b = math.inf, loc= 0, scale = sigma_s_current, size=1)[0]
        s_current += error
        if(i==0):
            print("s = ",s_current)
            print("sigma_s = ", sigma_s_current)
        x[i] = s_current * kappa[electricity_data.df.Day_type[i]] # on multiplie s_n par la valeur de kappa au jour n  
    return(x)

def x_heat_generation(n,temperature_data,u_heat, sigma_g_param, hour, len_initialization = 1000):
    """x_heat_generation : simulation of the x_heat according to the law described in the article"""
    #Initialization
    x = np.zeros(n)
    nu = truncnorm.rvs(a = 0, b = math.inf, loc= 0, scale = sigma_g_param, size=1)[0]
    sigma_g_current = nu
    error = truncnorm.rvs(a = -math.inf, b = 0, loc= 0, scale = sigma_g_current, size=1)[0]
    g_heat_current=error
    for i in range(1,len_initialization,1):
        nu = truncnorm.rvs(a = -sigma_g_current / sigma_g_param , b = math.inf, loc= 0, scale = sigma_g_param, size=1)[0]
        sigma_g_current += nu
        error = truncnorm.rvs(a = - math.inf , b = -g_heat_current / sigma_g_current , loc= 0, scale = sigma_g_current, size=1)[0]
        g_heat_current += error
    for i in range(0,n,1):
        nu = truncnorm.rvs(a = -sigma_g_current / sigma_g_param , b = math.inf, loc= 0, scale = sigma_g_param, size=1)[0]
        sigma_g_current += nu
        error = truncnorm.rvs(a = - math.inf , b = -g_heat_current / sigma_g_current , loc= 0, scale = sigma_g_current, size=1)[0]
        g_heat_current += error
        if(i==0):
            print("g_heat = ",g_heat_current)
            print("sigma_g = ", sigma_g_current)
        x[i] = g_heat_current * min(temperature_data.df.iloc[i, hour]-u_heat,0) # on multiplie s_n par la différence de la température 
    return(x)

def electricity_simulation(temperature_data, electricity_data, theta, hour, len_initialization = 1000):
    """electricity_simulation : function to be called in order to retrieve the electricity simulation
    We compute separately v (the main error), x_season and x_heat and we sum them at the end
    Parameters are :
     - an instance of the class for the temperature
     - an instance of the class for the electricity(in order to retrieve the day-type)
     - the vector theta from theta_generation
     - the index of the hour"""
    sigma2_s_param,sigma2_g_param,u_heat,kappa,sigma2 = theta
    n = len(temperature_data.df)
    v = np.random.normal(size = n, loc = 0, scale = sigma2**0.5) #loc = mean / scale = standard deviation /!\
    return(v + x_season_generation(n,electricity_data,kappa, sigma2_s_param**0.5, len_initialization = len_initialization ) + x_heat_generation(n,temperature_data,u_heat, sigma2_g_param**0.5, hour, len_initialization = len_initialization ) )