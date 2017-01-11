
# coding: utf-8

# In[ ]:

from scipy.stats import truncnorm
from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt

class Particle_Filtering_Model(object):
    def __init__(self,u_heat,kappa,sigma2, sigma2_s_param, sigma2_g_param, nb_particles, len_filtering, len_prediction = 5):
        """constructor
        u_heat,kappa,sigma2, sigma2_s_param, sigma2_g_param : float which define the fixed parameters of the model"""
        import numpy as np
        from scipy.stats import truncnorm
        import math
        #Define the fixed parameters theta of the model
        self.u_heat = u_heat
        self.kappa = kappa
        self.sigma2 = sigma2
        self.sigma2_s_param = sigma2_s_param
        self.sigma2_g_param = sigma2_g_param
        #Define the number of particles we want to simulate and the length of the filtering process (from date 0 to date n0)
        self.nb_particles = nb_particles
        self.len_filtering = len_filtering
        #Initialize the numpy arrays of x, s, g_heat, sigma_s and sigma_g
        self.s = np.zeros((len_filtering, nb_particles))
        self.sigma_s = np.zeros((len_filtering, nb_particles))
        self.sigma_g = np.zeros((len_filtering, nb_particles))
        self.g_heat = np.zeros((len_filtering, nb_particles))
        self.x = np.zeros((len_filtering, nb_particles))
        #Initialize the weights
        self.lw = np.zeros((len_filtering, nb_particles))
        self.w = np.zeros((len_filtering, nb_particles))
        self.log_likelihood = np.zeros(len_filtering) #/!\ This will not exactly represent the likelihood but only the likelihood up to a factor
        #Index_Resampling_Matrix
        self.index_resample = np.zeros((len_filtering, nb_particles))
        #self.index_resample[1,] = np.arange(nb_particles)
        #Initialize performance evaluation
        self.relative_error = np.zeros((len_filtering, len_prediction))
        self.confident_interval = np.zeros((len_filtering, len_prediction))
        #Prediction
        self.len_prediction = len_prediction
        self.prediction_value = np.zeros((len_filtering, len_prediction))
        self.prediction_lower = np.zeros((len_filtering, len_prediction))
        self.prediction_upper = np.zeros((len_filtering, len_prediction))
    
    
    def exp_and_normalize(self, lw):
        """Computes the normalized weights from the non normalized log weights
        lw : vector of the log weights (row numpy array)"""
        w = np.exp(lw - max(lw)) #np.exp Calculates the exponential of all elements in the input array.
        res = w / sum(w)
        return(res)
    
    def compute_log_g_y(self, x, elec):
        """ Computes the log of the likelihood function 
        x : float
        elec : float """
        res = -((elec-x)**2)/(2*self.sigma2)
        return(res)
    
    def vcompute_log_g_y(self, x_vector, elec):
        """ Computes the log of the likelihood function for all particles at the same time
        x : row of numpy array (vector of the values of x for each particle at the considered time)
        elec : float (value of the observed electricty load y at the considered time) """
        vectorized_function = np.vectorize(self.compute_log_g_y, excluded=['elec'])
        return(vectorized_function(x_vector, elec))
    
    def compute_x(self, s, g_heat, temperature, daytype):
        """ Computes the value of x for each particle at a specific time t
        Parameters :
        -s are g_heat are row vectors (value of s and g_heat for each particle)
        -temperature and daytype are constants (observed data at time t)
        Returns a row vector"""
        x_season = self.kappa[daytype]*s
        if temperature < self.u_heat :
            x_heat = (temperature-self.u_heat)*g_heat
        else :
            x_heat = np.zeros(g_heat.shape)
        x = x_season + x_heat
        return(x)
    
    def initialization_SIS_withoutMCMC(self, temperature_ts, daytype_ts):
        """ Initialization of the values of s, g_heat, sigma_s and sigma_g (and hence x) using the prior distributions of the article
        This initialization could also be done using a first MCMC Gibbs sampling step"""
        self.s[0,] = truncnorm.rvs(a = 0,b = math.inf, loc= 0, scale = 10**4, size=self.nb_particles)#loc = mean / scale = standard deviation /!\
        self.sigma_s[0,] =(1 / (np.random.gamma(size = self.nb_particles, shape = 0.01, scale = 100)))**0.5
        self.sigma_g[0,] = (1 / (np.random.gamma(size = self.nb_particles, shape = 0.01, scale = 100)))**0.5
        self.g_heat[0,] = truncnorm.rvs(a = - math.inf,b = 0, loc= 0, scale = 10**4, size=self.nb_particles)#loc = mean / scale = standard deviation /!\
        self.x[0,] = self.compute_x(s = self.s[0,], g_heat = self.g_heat[0,], temperature = temperature_ts[0], daytype = daytype_ts[0])
        
    def initialization_SIS_withParam(self, temperature_ts, daytype_ts, param):
        """ Initialization using parameters chosen by the user"""
        self.s[0,] = param[0] + np.zeros(self.nb_particles)
        self.g_heat[0,] = param[1] + np.zeros(self.nb_particles)
        self.sigma_s[0,] = param[2] + np.zeros(self.nb_particles)
        self.sigma_g[0,] = param[3] + np.zeros(self.nb_particles)
        self.x[0,] = self.compute_x(s = self.s[0,], g_heat = self.g_heat[0,], temperature = temperature_ts[0], daytype = daytype_ts[0])
            
        
            
    
    def sample_new_sigma_g(self, sigma_g_prev):
        """ Simulates the current sigma_g given the previous sigma_g
        Parameters :
        -sigma_g_prev : float 
        Return a float"""
        #see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        #/!\ The standard form of this distribution is a STANDARD normal truncated to the range [a, b] — 
        #notice that a and b are defined over the domain of the STANDARD normal. 
        #To convert clip values for a specific mean and standard deviation :
        my_mean = 0
        my_std = (self.sigma2_g_param)**0.5
        myclip_a = - sigma_g_prev
        myclip_b = math.inf
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        error = truncnorm.rvs(a = a, b = b, loc= my_mean, scale = my_std, size=1)[0]#loc = mean / scale = standard deviation
        res = sigma_g_prev + error
        return(res)
    
    def vsample_new_sigma_g(self, sigma_g_prev_vector):
        """ Simulates the current sigma_g for each particle given the previous sigma_g for each particle
        Parameters : 
        -sigma_g_prev : vector (row of a numpy array)
        Returns a row vector"""
        vectorized_function = np.vectorize(self.sample_new_sigma_g)
        return(vectorized_function(sigma_g_prev_vector))
    
    def sample_new_sigma_s(self, sigma_s_prev):
        """ Simulates the current sigma_s given the previous sigma_s
        Parameters : 
        -sigma_s_prev : float 
        Returns a float """
        my_mean = 0
        my_std = (self.sigma2_s_param)**0.5
        myclip_a = - sigma_s_prev
        myclip_b = math.inf
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        error = truncnorm.rvs(a = a, b = b, loc= my_mean, scale = my_std, size=1)[0]#loc = mean / scale = standard deviation
        res = sigma_s_prev + error
        return(res)
    
    def vsample_new_sigma_s(self, sigma_s_prev_vector) :
        """ Simulates the current sigma_s for each particle given the previous sigma_s for each particle
        Parameters : 
        -sigma_s_prev : vector (row of a numpy array) 
        Returns a row vector """
        vectorized_function = np.vectorize(self.sample_new_sigma_s)
        return(vectorized_function(sigma_s_prev_vector))
    
    def sample_new_g_heat(self, g_heat_prev, sigma_g_current):
        """ Simulates the current g_heat given the previous g_heat and the current sigma_g
        Parameters :
        -g_heat_prev : float
        -sigma_g_current : float
        Returns a float """
        my_mean = 0
        my_std = sigma_g_current
        myclip_a = - math.inf
        myclip_b = -g_heat_prev
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        error = truncnorm.rvs(a = a, b = b, loc= my_mean, scale = my_std, size=1)#loc = mean / scale = standard deviation
        res = g_heat_prev + error
        return(res)
    
    def vsample_new_g_heat(self, g_heat_prev, sigma_g_current):
        """ Simulates the current g_heat for each particle given the previous g_heat for each particle and the current 
        sigma_g for each particle.
        Parameters :
        -g_heat_prev :row of a numpy array (size =  1 x nb_particles)
        -sigma_g_current : row of a numpy array (size = 1 x nb_particles)
        Returns a row vector"""
        nb_part = len(g_heat_prev)
        g_heat_new =  np.zeros((1, nb_part))
        for j in range(nb_part):
            g_heat_new[0,j] = self.sample_new_g_heat(g_heat_prev[j], sigma_g_current[j])
        return(g_heat_new)
    
    def sample_new_s(self, s_prev, sigma_s_current):
        """ Simulates the current s given the previous s and the current sigma_s
        Parameters :
        -s_prev : float
        -sigma_s_current : float"""
        my_mean = 0
        my_std = sigma_s_current
        myclip_a = -s_prev
        myclip_b = math.inf
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        error = truncnorm.rvs(a = a, b = b, loc= my_mean, scale = my_std, size=1)#loc = mean / scale = standard deviation
        res = s_prev + error
        return(res)
    
    def vsample_new_s(self, s_prev, sigma_s_current):
        """ Simulates the current s for each particle given the previous s for each particle and the current 
        sigma_s for each particle.
        Parameters :
        -s_prev :row of a numpy array (size =  1 x nb_particles)
        -sigma_s_current : row of a numpy array (size = 1 x nb_particles)"""
        nb_part = len(s_prev)
        s_new =  np.zeros((1, nb_part))
        for j in range(nb_part):
            s_new[0,j] = self.sample_new_s(s_prev[j], sigma_s_current[j])
        return(s_new)
    
    def sample_from_transition(self, s_prev, g_heat_prev, sigma_s_prev, sigma_g_prev):
        """ Samples the vector x, s, g_heat, sigma2_s, sigma2_g at time n from transition density
        Given s, g_heat, sigma2_s, sigma2_g at time (n-1)
        Parameters :
        s_prev, g_heat_prev, sigma2_s_prev, sigma2_g_prev : row of numpy array (size = 1 x nb_particles)"""
        sigma_g_current = self.vsample_new_sigma_g(sigma_g_prev)
        sigma_s_current = self.vsample_new_sigma_s(sigma_s_prev)
        g_heat_current = self.vsample_new_g_heat(g_heat_prev, sigma_g_current)
        s_current = self.vsample_new_s(s_prev, sigma_s_current)
        return(s_current, g_heat_current, sigma_s_current, sigma_g_current)
    
    def initialization_SIS_withMCMC(self, temperature_ts, daytype_ts, elec_ts, len_initialization_MCMC):
        """ Initialization of the Particle Filter at the date (len_initialization_MCMC - 1) 
        Using a Gibbs sampler to simulate from the smoothed distribution between date 0 and date (len_initialization_MCMC - 1)
        Note that we assume here that sigma_s and sigma_g do not follow any dynamic (they remain fixed to sigma_s_star and sigma_g_star), as suggested in the article.
        Hence, we add at the end of the Gibbs sampling process a prior on sigma_s and sigma_g at date (len_initialization_MCMC - 1) using the empirical errors epsilon observed in the Gibbs sampling to choose the parameters of this prior."""
        ###1. Create variables to store the MCMC results between date = 0 and date = len_initialization_MCMC - 1
        #(Convergence can therefore be checked by the user)
        self.MCMC_s = np.zeros((len_initialization_MCMC, self.nb_particles)) 
        self.MCMC_g_heat = np.zeros((len_initialization_MCMC, self.nb_particles))
        self.MCMC_sigma2_s_star = np.zeros((1, self.nb_particles)) #See the article: we assume here that sigma_s and sigma_g do not follow any dynamic (they remain fixed to sigma_s_star and sigma_g_star)
        self.MCMC_sigma2_g_star = np.zeros((1, self.nb_particles))
        ###2. Gibbs-sampler 
        #Gibbs Initialization
        self.MCMC_s[0,0] = truncnorm.rvs(a = 0,b = math.inf, loc= 0, scale = 10**4, size=1)[0]
        self.MCMC_g_heat[0,0] =  truncnorm.rvs(a = - math.inf,b = 0, loc= 0, scale = 10**4, size=1)[0]
        self.MCMC_sigma2_s_star[0, 0] = 1 / (np.random.gamma(size = 1, shape = 0.01, scale = 100)[0])
        self.MCMC_sigma2_g_star[0, 0] = 1 / (np.random.gamma(size = 1, shape = 0.01, scale = 100)[0])
        for i in range(1, len_initialization_MCMC,1):
            self.MCMC_s[i,0] = self.sample_new_s(s_prev = self.MCMC_s[i-1,0],  sigma_s_current = (self.MCMC_sigma2_s_star[0, 0])**0.5)
            self.MCMC_g_heat[i,0] = self.sample_new_g_heat(g_heat_prev = self.MCMC_g_heat[i-1,0],  sigma_g_current = (self.MCMC_sigma2_g_star[0, 0])**0.5)
        #Gibbs Main
        for j in range(1, self.nb_particles, 1):
            self.MCMC_s[:,j] = self.MCMC_s[:,j-1]
            self.MCMC_g_heat[:,j] = self.MCMC_g_heat[:,j-1]
            self.MCMC_sigma2_s_star[:,j] = self.MCMC_sigma2_s_star[:,j-1]
            self.MCMC_sigma2_g_star[:,j] = self.MCMC_sigma2_g_star[:,j-1]
            ##a. Simulate s0
            #Compute the denominator of the variance and the mean
            denom_s_0 = (10**8)*self.MCMC_sigma2_s_star[0, j]*(self.kappa[daytype_ts[0]]**2) + self.sigma2*self.MCMC_sigma2_s_star[0, j] + (10**8)*self.sigma2 
            #Compute the numerator of the mean
            if (self.u_heat > temperature_ts[0]):
                numerator_mean_s_0 = (10**8)*self.sigma2* self.MCMC_s[1,j] + (10**8)*self.MCMC_sigma2_s_star[0, j]*self.kappa[daytype_ts[0]]*(elec_ts[0] - self.MCMC_g_heat[0,j]*(temperature_ts[0] - self.u_heat))
            else :
                numerator_mean_s_0 = (10**8)*self.sigma2* self.MCMC_s[1,j] + (10**8)*self.MCMC_sigma2_s_star[0, j]*self.kappa[daytype_ts[0]]*(elec_ts[0])
            #Compute the final parameters of the truncated normal that simulates from the full conditional of s_0
            mean_s_0 = numerator_mean_s_0 / denom_s_0
            var_s_0 = ((10**8) *self.sigma2*self.MCMC_sigma2_s_star[0, j]) / denom_s_0
            a_s_0, b_s_0 = (0 - mean_s_0) /  (var_s_0**0.5), math.inf
            self.MCMC_s[0,j] = truncnorm.rvs(a = a_s_0, b = b_s_0, loc= mean_s_0, scale = (var_s_0**0.5), size=1)[0]#loc = mean / scale = standard deviation
            if(self.MCMC_s[0,j]==math.inf):
                self.MCMC_s[0,j]=0
            ##b. Simulate all other s(i)
            for i in range(1, len_initialization_MCMC,1):
                denom_s_i = 2*self.sigma2 + self.MCMC_sigma2_s_star[0, j]*(self.kappa[daytype_ts[i]]**2)
                if (i+1 < len_initialization_MCMC-1):
                    dependence_next_s = self.MCMC_s[i+1,j]
                else : 
                    dependence_next_s = 0
                #Compute the numerator of the mean
                if (self.u_heat > temperature_ts[i]):
                    numerator_mean_s_i = self.sigma2*(self.MCMC_s[i-1,j] + dependence_next_s) + self.MCMC_sigma2_s_star[0, j]*self.kappa[daytype_ts[i]]*(elec_ts[i] - self.MCMC_g_heat[i,j]*(temperature_ts[i] - self.u_heat))
                else :
                    numerator_mean_s_i = self.sigma2*(self.MCMC_s[i-1,j] + dependence_next_s) + self.MCMC_sigma2_s_star[0, j]*self.kappa[daytype_ts[i]]*(elec_ts[i])
                mean_s_i = numerator_mean_s_i / denom_s_i
                var_s_i = (self.sigma2*self.MCMC_sigma2_s_star[0, j]) / denom_s_i
                a_s_i, b_s_i = (0 - mean_s_i) /  (var_s_i**0.5), math.inf
                self.MCMC_s[i,j] = truncnorm.rvs(a = a_s_i, b = b_s_i, loc= mean_s_i, scale = (var_s_i**0.5), size=1)[0]
                if(self.MCMC_s[i,j]==math.inf):
                    self.MCMC_s[i,j]=0
            ##c. Simulate g_heat0
            #Compute the numerator and the denominator of the mean
            if (self.u_heat > temperature_ts[0]):
                denom_g_0 = (10**8)*self.MCMC_sigma2_g_star[0, j]*((temperature_ts[0] - self.u_heat )**2) + self.sigma2*self.MCMC_sigma2_g_star[0, j] + (10**8)*self.sigma2 
                numerator_mean_g_0 = (10**8)*self.sigma2* self.MCMC_g_heat[1,j] + (10**8)*self.MCMC_sigma2_g_star[0, j]*(temperature_ts[0] - self.u_heat)*(elec_ts[0] - self.MCMC_s[0,j]*self.kappa[daytype_ts[0]])
            else :
                denom_g_0 = self.sigma2*self.MCMC_sigma2_g_star[0, j] + (10**8)*self.sigma2 
                numerator_mean_g_0 = (10**8)*self.sigma2* self.MCMC_g_heat[1,j]
            #Compute the final parameters of the truncated normal that simulates from the full conditional of g_0
            mean_g_0 = numerator_mean_g_0 / denom_g_0
            var_g_0 = ((10**8) *self.sigma2*self.MCMC_sigma2_g_star[0, j]) / denom_g_0
            a_g_0, b_g_0 = -math.inf, (0 - mean_g_0) /  (var_g_0**0.5)
            self.MCMC_g_heat[0,j] =  truncnorm.rvs(a = a_g_0, b = b_g_0, loc= mean_g_0, scale = (var_g_0**0.5), size=1)[0]
            if(self.MCMC_g_heat[0,j]==-math.inf):
                self.MCMC_g_heat[0,j]=0
            ##d. Simulate all other g_heat(i)
            for i in range(1, len_initialization_MCMC,1):
                if (i+1 < len_initialization_MCMC-1):
                    dependence_next_g = self.MCMC_g_heat[i+1,j]
                else : 
                    dependence_next_g = 0
                if (self.u_heat > temperature_ts[i]):
                    denom_g_i = 2*self.sigma2 + self.MCMC_sigma2_g_star[0, j]*((temperature_ts[i] - self.u_heat )**2)
                    numerator_mean_g_i = self.sigma2*(self.MCMC_g_heat[i-1,j] + dependence_next_g) + self.MCMC_sigma2_g_star[0, j]*(temperature_ts[i] - self.u_heat )*(elec_ts[i] - self.MCMC_s[i,j]*self.kappa[daytype_ts[i]])
                else :
                    denom_g_i = 2*self.sigma2
                    numerator_mean_g_i = self.sigma2*(self.MCMC_g_heat[i-1,j] + dependence_next_g)
                mean_g_i = numerator_mean_g_i / denom_g_i
                var_g_i = (self.sigma2*self.MCMC_sigma2_g_star[0, j]) / denom_g_i
                a_g_i, b_g_i = -math.inf, (0 - mean_g_i) /  (var_g_i**0.5)
                self.MCMC_g_heat[i,j] =  truncnorm.rvs(a = a_g_i, b = b_g_i, loc= mean_g_i, scale = (var_g_i**0.5), size=1)[0]
                if(self.MCMC_g_heat[i,j]==-math.inf):
                    self.MCMC_g_heat[i,j]=0
            ##e. Simulate the variances
            shape_variances = 0.01 + ((len_initialization_MCMC - 1)/2)
            s_lag = np.roll(self.MCMC_s[:,j], 1)
            s_lag[0] = self.MCMC_s[0,j]
            rate_s = 0.01 + sum((self.MCMC_s[:,j] - s_lag)**2)
            self.MCMC_sigma2_s_star[0, j] = 1 / (np.random.gamma(size = 1, shape = shape_variances, scale = 1/rate_s )[0])
            g_lag = np.roll(self.MCMC_g_heat[:,j], 1)
            g_lag[0] = self.MCMC_g_heat[0,j]
            rate_g = 0.01 + sum((self.MCMC_g_heat[:,j] - g_lag)**2)
            self.MCMC_sigma2_g_star[0, j] = 1 / (np.random.gamma(size = 1, shape = shape_variances, scale = 1/rate_g )[0])
        ###3. Add a prior on sigma_s and sigma_g based on the estimated errors
        s_lag = np.roll(self.MCMC_s, shift = 1, axis =0)
        s_lag[0,] = self.MCMC_s[0,:]
        error_s = self.MCMC_s - s_lag 
        error_s = error_s[1:len_initialization_MCMC, :] #delete first row
        #compute the standard error at each time i
        error_s_std = np.std(error_s, axis =1)
        #compute the average of the standard errors among all times
        error_s_mean_of_std = np.mean(error_s_std)
        error_s_std_of_std = np.std(error_s_std)
        #same process for g_heat
        g_lag = np.roll(self.MCMC_g_heat, shift = 1, axis =0)
        g_lag[0,] = self.MCMC_g_heat[0,:]
        error_g = self.MCMC_g_heat - g_lag 
        error_g = error_g[1:len_initialization_MCMC, :] #delete first row
        error_g_std = np.std(error_g, axis =1)
        error_g_mean_of_std = np.mean(error_g_std)
        error_g_std_of_std = np.std(error_g_std)
        ###4. Return the initialization of the Particle Filter at date (len_initialization_MCMC - 1)
        self.s[0,] = self.MCMC_s[len_initialization_MCMC-1,self.nb_particles-1] + np.zeros(self.nb_particles)
        self.g_heat[0,] = self.MCMC_g_heat[len_initialization_MCMC-1,self.nb_particles-1]+ np.zeros(self.nb_particles)
        self.sigma_s[0,] = self.MCMC_sigma2_s_star[0,self.nb_particles-1]**0.5+ np.zeros(self.nb_particles)
        self.sigma_g[0,] = self.MCMC_sigma2_g_star[0,self.nb_particles-1]**0.5+ np.zeros(self.nb_particles)
        #self.sigma_s[0,] = truncnorm.rvs(a = (- error_s_mean_of_std / error_s_std_of_std) ,b = math.inf, loc= error_s_mean_of_std, scale = error_s_std_of_std, size=self.nb_particles)
        #self.sigma_g[0,] = truncnorm.rvs(a = (- error_g_mean_of_std / error_g_std_of_std) ,b = math.inf, loc= error_g_mean_of_std, scale = error_g_std_of_std, size=self.nb_particles)
        self.x[0,] = self.compute_x(s = self.s[0,], g_heat = self.g_heat[0,],  temperature = temperature_ts[len_initialization_MCMC-1], daytype = daytype_ts[len_initialization_MCMC-1])
       
    def ESS_calc(self, n):
        """This function calculates the ESS, which we will use to alleviate the degeneracy.
        - w is the matrix of weights
        - n is the the time of the filtering
        """
        res = 1 / sum (self.w[n,]**2)
        return(res) 
    
    def Resample(self,n):
        """This function allows us to reasample our weight to only keep the observations with important weights
        which are more reprenstative of targeted distribution.
        """   
        multinomial = np.random.multinomial(1,self.w[n,],self.nb_particles)
        new_x = np.zeros(self.nb_particles)
        new_s = np.zeros(self.nb_particles)
        new_g_heat = np.zeros(self.nb_particles)
        new_sigma_s = np.zeros(self.nb_particles)
        new_sigma_g = np.zeros(self.nb_particles)
        for i in range(self.nb_particles):
            new_x[i]=self.x[n,np.argmax(multinomial[i,])]
            new_s[i]=self.s[n,np.argmax(multinomial[i,])]
            new_g_heat[i]=self.g_heat[n,np.argmax(multinomial[i,])]
            new_sigma_s[i]=self.sigma_s[n,np.argmax(multinomial[i,])]
            new_sigma_g[i]=self.sigma_g[n,np.argmax(multinomial[i,])]
            self.index_resample[n,i] = np.argmax(multinomial[i,])
            
        self.x[n,] = new_x
        self.s[n,] = new_s
        self.g_heat[n,] = new_g_heat
        self.sigma_s[n,] = new_sigma_s
        self.sigma_g[n,] = new_sigma_g
        self.w[n,] = 1/self.nb_particles + np.zeros(self.nb_particles)
         
        
    def Performance_Evaluation(self, date_start, elec_ts):
        """This function evaluates the efficient of the prediction. It computes before the resampling. inf_bound and sup_bound are the bounds of the confident interval"""
        for n in range(self.len_filtering):
            for d in range(self.len_prediction):
                self.relative_error[n,d] = abs(self.prediction_value[n,d] - elec_ts[date_start + n + d]) / elec_ts[date_start + n + d]
                if( elec_ts[date_start + n + d] >= self.prediction_lower[n,d] and elec_ts[date_start + n + d] <= self.prediction_upper[n,d]):
                    self.confident_interval[n,d] = 1
        
    def Prediction(self, n, index_data, temperature_ts, daytype_ts, inf_bound = 2.5, sup_bound = 97.5 ):
        """This function computes a predictio value (until 5 days like the article). We assume that the temperature is known (it can be estimated by weather institute)."""
        #For the prediction (n-1) --> n
        self.prediction_value[n,0] = np.mean(self.x[n,])
        self.prediction_lower[n,0] = np.percentile(self.x[n,], inf_bound)
        self.prediction_upper[n,0] = np.percentile(self.x[n,], sup_bound)
        #For the prediction (n-1) --> (n+1)
        if( self.len_prediction > 0 ):
            temp_s, temp_g_heat, temp_sigma_s, temp_sigma_g = self.sample_from_transition(s_prev = self.s[n,], 
                                g_heat_prev = self.g_heat[n,], sigma_s_prev = self.sigma_s[n,], 
                                sigma_g_prev = self.sigma_g[n,])
            temp_x = self.compute_x(s = temp_s, g_heat = temp_g_heat,  temperature = temperature_ts[index_data + 1], daytype = daytype_ts[index_data+1])
            self.prediction_value[n,1] = np.mean(temp_x)
            self.prediction_lower[n,1] = np.percentile(temp_x, inf_bound)
            self.prediction_upper[n,1] = np.percentile(temp_x, sup_bound)
        #For the prediction (n-1) --> (n+i),i >1    
        for i in range(2, self.len_prediction):
            temp_s, temp_g_heat, temp_sigma_s, temp_sigma_g = self.sample_from_transition(s_prev = temp_s[0], 
                                g_heat_prev = temp_g_heat[0], sigma_s_prev = temp_sigma_s, 
                                sigma_g_prev = temp_sigma_g)

            temp_x = self.compute_x(s = temp_s, g_heat = temp_g_heat,  temperature = temperature_ts[index_data + i], daytype = daytype_ts[index_data+i])
            self.prediction_value[n,i] = np.mean(temp_x)
            self.prediction_lower[n,i] = np.percentile(temp_x, inf_bound)
            self.prediction_upper[n,i] = np.percentile(temp_x, sup_bound)
    
        
    
    def SIS_filter(self, elec_ts, temperature_ts, daytype_ts, MCMC_init = False, len_initialization_MCMC = 0, Resampling = True,
                  withParam = False, param = np.zeros(4), compute_prediction = False):
        """ Sequential Importance Sampling for filtering
        Given the set of parameters, simulate particles for each day until today (past)"""
        if (MCMC_init):
            #Initialization : time n = 0 
            connected = False
            while not connected:
                try:
                    self.initialization_SIS_withMCMC(temperature_ts, daytype_ts, elec_ts, len_initialization_MCMC)
                    connected = True
                except :
                    pass
            date_start = len_initialization_MCMC-1
        elif(withParam) :
            self.initialization_SIS_withParam(temperature_ts, daytype_ts, param )
            date_start = 0
        else:
            self.initialization_SIS_withoutMCMC(temperature_ts, daytype_ts)
            date_start = 0
        self.lw[0,] = self.vcompute_log_g_y(x_vector = self.x[0,], elec = elec_ts[date_start]) #Compute the log weights at time 0
        self.log_likelihood[0] = np.log(sum(np.exp(self.lw[0,])))
        self.w[0,] = self.exp_and_normalize(self.lw[0,])#Compute the normalized weights at time 0
        self.lw[0,] = np.log(self.w[0,]) #Compute the log of the normalized weights at time 0
        if( compute_prediction):
            self.Prediction(0, date_start, temperature_ts, daytype_ts)
        #Main : at time n > 0
        for n in range(1,self.len_filtering,1):
            #1. Sample from transition density
            self.s[n,], self.g_heat[n,], self.sigma_s[n,], self.sigma_g[n,] = self.sample_from_transition(s_prev = self.s[(n-1),], 
                                g_heat_prev = self.g_heat[(n-1),], sigma_s_prev = self.sigma_s[(n-1),], 
                                sigma_g_prev = self.sigma_g[(n-1),])
            self.x[n,] = self.compute_x(s = self.s[n,], g_heat = self.g_heat[n,],  temperature = temperature_ts[date_start + n], daytype = daytype_ts[date_start + n])
            #2. Update the weights
            #self.lw[n,] = self.lw[(n-1),] + self.vcompute_log_g_y(x_vector = self.x[n,], elec = elec_ts[date_start + n]) #Compute the log weights at time n
            log_g_y_temp = self.vcompute_log_g_y(x_vector = self.x[n,], elec = elec_ts[date_start + n])
            #print(np.log(sum(np.exp(log_g_y_temp))))
            self.lw[n,] = self.lw[(n-1),] + log_g_y_temp #Compute the log weights at time n
            self.log_likelihood[n] = self.log_likelihood[(n-1)] + np.log(sum(np.exp(log_g_y_temp))) #Compute the likelihood at time n
            self.w[n,] = self.exp_and_normalize(self.lw[n,])#Compute the normalized weights at time 0
            
            #Prediction
            if( compute_prediction):
                self.Prediction(n, date_start + n, temperature_ts, daytype_ts)
            
            #Resample
            if( Resampling ):
                if (self.ESS_calc(n) < 0.001*self.nb_particles):
                    #print("ESS below 0.001 for ",n)
                    self.w[n,] =  1/self.nb_particles + np.zeros(self.nb_particles)     
                elif (self.ESS_calc(n) < 0.5*self.nb_particles and self.ESS_calc(n) >= 0.001*self.nb_particles):
                    #print("Resampling for ",n)
                    self.Resample(n) 

            self.lw[n,] = np.log(self.w[n,]) #Compute the log of the normalized weights at time 0
            
        for i in range(self.index_resample.shape[0]):
            if (np.sum(self.index_resample[i,])==0):
                self.index_resample[i,]=self.index_resample[i-1,]

        #Performance Evaluation
        if( compute_prediction):
            self.Performance_Evaluation(date_start, elec_ts)
        


# In[ ]:

class PMCMC(object):
    """ Particle MCMC algorithm in order to estimate the parameters of the model"""
    def __init__(self, u_heat,kappa,sigma2, sigma2_s_param, sigma2_g_param, nb_particles, len_filtering, len_metropolis):
        """ Initialize the parameters to estimate (u_heat, kappa,sigma2, sigma2_s_param, sigma2_g_param)
        And give the number of particles and the length of the filtering process which will be used in each Particle Filter (in each step of the Metropolis-Hastings algorithm)"""
        #Initialize the parameters of the Particle Filters
        #Particle_Filtering_Model.__init__(self, u_heat,kappa,sigma2, sigma2_s_param, sigma2_g_param, nb_particles, len_filtering)
        #Store the number of iterations in the Metropolis-Hastings algorithm
        self.len_metropolis = len_metropolis
        self.nb_particles = nb_particles
        self.len_filtering = len_filtering
        #Initialize the vector representing the evolution of each parameter in the Metropolis-Hastings algorithm
        self.u_heat_evolution = np.zeros(self.len_metropolis)
        self.u_heat_evolution[0] = u_heat
        self.kappa_evolution = np.zeros((self.len_metropolis, len(kappa)))
        self.kappa_evolution[0,] = kappa
        self.sigma2_evolution = np.zeros(self.len_metropolis)
        self.sigma2_evolution[0] = sigma2
        self.sigma2_s_param_evolution = np.zeros(self.len_metropolis)
        self.sigma2_s_param_evolution[0] = sigma2_s_param
        self.sigma2_g_param_evolution = np.zeros(self.len_metropolis)
        self.sigma2_g_param_evolution[0] = sigma2_g_param
    
    def log_prior_density(self, u_heat, sigma2, sigma2_s_param, sigma2_g_param, number_of_daytypes):
        """ Compute the value of the prior density at point theta =  u_heat,kappa,sigma2, sigma2_s_param, sigma2_g_param.
        The prior densities are given by the article p20 :
        -u_heat : Normal(mean = 14, variance = 1)
        -kappa/number_of_daytypes : Dirichlet(1,...,1) => since this distribution is uniform (constant pdf), we can omit it in the calculus because it will disappear in the ratio
        -sigma2 : InverseGamma(0.01, 0.01)
        -sigma2_s_param : InverseGamma(0.01, 0.01)
        -sigma2_g_param: InverseGamma(0.01, 0.01) 
        """
        result = -((u_heat-14)**2)/2 #Prior on u_heat
        result = result + (-0.01-1)*np.log(sigma2) - (0.01/sigma2) #Prior on sigma2
        result = result + (-0.01-1)*np.log(sigma2_s_param) - (0.01/sigma2_s_param) #Prior on sigma2_s_param
        result = result + (-0.01-1)*np.log(sigma2_g_param) - (0.01/sigma2_g_param) #Prior on sigma2_g_param
        return(result)
    
    def PMMH(self, elec_ts, temperature_ts, daytype_ts, std_u_heat_proposal, std_sigma2_proposal, 
             std_sigma2_s_param_proposal, std_sigma2_g_param_proposal, MCMC_init = False, len_initialization_MCMC = 0, Resampling = True,
             withParam = False, param = np.zeros(4)):
        """Particle Marginal Metropolis-Hastings
        Goal : The PMMH algorithm is an MCMC algorithm which targets the full joint posterior distribution p(theta, x |y)."""
        number_of_daytypes = len(set(daytype_ts))
        #Initialize by running a PF on init parameters
        PF = Particle_Filtering_Model(self.u_heat_evolution[0],self.kappa_evolution[0,],self.sigma2_evolution[0], self.sigma2_s_param_evolution[0], 
                                          self.sigma2_g_param_evolution[0], self.nb_particles, self.len_filtering)
        PF.SIS_filter(elec_ts, temperature_ts, daytype_ts, MCMC_init = MCMC_init, len_initialization_MCMC = len_initialization_MCMC, Resampling =Resampling,
                  withParam = withParam, param = param, compute_prediction = False)
        log_likelihood_temp = PF.log_likelihood[self.len_filtering-1] #Initialize the likelihood
        print("1st log likelihood = ", log_likelihood_temp)
        log_prior_temp = self.log_prior_density(u_heat = self.u_heat_evolution[0], 
                                                sigma2 = self.sigma2_evolution[0], 
                                                sigma2_s_param = self.sigma2_s_param_evolution[0], 
                                                sigma2_g_param = self.sigma2_g_param_evolution[0], 
                                                number_of_daytypes = number_of_daytypes)
        print("1st prior =", log_prior_temp)
        acceptance_rate = 0
        for step in range((self.len_metropolis-1)):
            if (step % 30 == 0):
                np.savetxt('results_pmmh_u_heat.txt', self.u_heat_evolution, newline=';', delimiter = ',')
                np.savetxt('results_pmmh_sigma2.txt', self.sigma2_evolution, newline=';',delimiter = ',')
                np.savetxt('results_pmmh_sigma2_s_param.txt', self.sigma2_s_param_evolution,newline=';', delimiter = ',')
                np.savetxt('results_pmmh_sigma2_g_param.txt', self.sigma2_g_param_evolution, newline=';', delimiter = ',')
            #1. Sample argument proposals
            u_heat_proposal = np.random.normal(loc = self.u_heat_evolution[step], scale = std_u_heat_proposal)
            kappa_proposal = self.kappa_evolution[0,]
            #kappa_proposal = (np.random.dirichlet(size = 1, alpha = [1]*number_of_daytypes)[0])*number_of_daytypes #independent proposal
            my_mean = self.sigma2_evolution[step]
            my_std = std_sigma2_proposal
            myclip_a = 0
            myclip_b = math.inf
            a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            sigma2_proposal = truncnorm.rvs(a = a, b = b, loc= my_mean, scale = my_std, size=1)
            my_mean = self.sigma2_s_param_evolution[step]
            my_std = std_sigma2_s_param_proposal
            myclip_a = 0
            myclip_b = math.inf
            a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            sigma2_s_param_proposal =  truncnorm.rvs(a = a, b = b, loc= my_mean, scale = my_std, size=1)
            my_mean = self.sigma2_g_param_evolution[step]
            my_std = std_sigma2_g_param_proposal
            myclip_a = 0
            myclip_b = math.inf
            a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            sigma2_g_param_proposal =  truncnorm.rvs(a = a, b = b, loc= my_mean, scale = my_std, size=1)
            #2. Run a Particle Filter to obtain an estimation of the likelihood for the proposal
            PF = Particle_Filtering_Model(u_heat_proposal,kappa_proposal,sigma2_proposal, sigma2_s_param_proposal, 
                                          sigma2_g_param_proposal, self.nb_particles, self.len_filtering, compute_prediction = False)
            connected = False
            nb_try=0
            while ((not connected) and (nb_try < 5)):
                try:
                    PF.SIS_filter(elec_ts, temperature_ts, daytype_ts, MCMC_init = MCMC_init, len_initialization_MCMC = len_initialization_MCMC, Resampling =Resampling,
                                  withParam = withParam, param = param, compute_prediction = False)
                    connected = True
                except :
                    nb_try = nb_try+1
            
            if (nb_try==5):
                PF.SIS_filter(elec_ts, temperature_ts, daytype_ts, MCMC_init = MCMC_init, len_initialization_MCMC = len_initialization_MCMC, Resampling =Resampling,
                                  withParam = withParam, param = param, compute_prediction = False)
            #3. Compute acceptance probability ratio
            print("The log likelihood of the proposal", step," is : ", PF.log_likelihood[self.len_filtering-1])
            log_ratio = PF.log_likelihood[self.len_filtering-1] - log_likelihood_temp + (self.len_filtering/2)*(np.log(self.sigma2_evolution[step]) - np.log(sigma2_proposal))#likelihood ratio
            log_prior_proposal = self.log_prior_density(u_heat_proposal,sigma2_proposal, sigma2_s_param_proposal, sigma2_g_param_proposal, number_of_daytypes)
            print("The log prior of the proposal", step," is : ", log_prior_proposal)
            log_ratio = log_ratio + log_prior_proposal - log_prior_temp
            #Take the truncated normal proposals into account => the proposal is not exactly symmetric
            #See https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
            log_ratio = log_ratio + np.log(norm.cdf(self.sigma2_evolution[step]/std_sigma2_proposal, loc = 0, scale = 1)) - np.log(norm.cdf(sigma2_proposal/std_sigma2_proposal, loc = 0, scale = 1))
            log_ratio = log_ratio + np.log(norm.cdf(self.sigma2_s_param_evolution[step]/std_sigma2_s_param_proposal, loc = 0, scale = 1)) - np.log(norm.cdf(sigma2_s_param_proposal/std_sigma2_s_param_proposal, loc = 0, scale = 1))
            log_ratio = log_ratio + np.log(norm.cdf(self.sigma2_g_param_evolution[step]/std_sigma2_g_param_proposal, loc = 0, scale = 1)) - np.log(norm.cdf(sigma2_g_param_proposal/std_sigma2_g_param_proposal, loc = 0, scale = 1))
            print('The log ratio is : ', log_ratio)
            #4. With probability min(1, ratio), accept the proposal and store it in the corresponding argument vector
            u = np.random.uniform(low=0.0, high=1.0)
            print(np.log(u))
            if (np.log(u) < min(0, log_ratio)):
                acceptance_rate = acceptance_rate + 1
                log_likelihood_temp = PF.log_likelihood[self.len_filtering-1] #store the likelihood for next step
                log_prior_temp = log_prior_proposal #store the prior density value for the next step
                self.u_heat_evolution[step+1] = u_heat_proposal
                self.kappa_evolution[step+1,] = kappa_proposal
                self.sigma2_evolution[step+1] = sigma2_proposal
                self.sigma2_s_param_evolution[step+1] = sigma2_s_param_proposal
                self.sigma2_g_param_evolution[step+1] = sigma2_g_param_proposal
            else :
                self.u_heat_evolution[step+1] = self.u_heat_evolution[step]
                self.kappa_evolution[step+1,] = self.kappa_evolution[step,]
                self.sigma2_evolution[step+1] = self.sigma2_evolution[step]
                self.sigma2_s_param_evolution[step+1] = self.sigma2_s_param_evolution[step] 
                self.sigma2_g_param_evolution[step+1] = self.sigma2_g_param_evolution[step]
        acceptance_rate = (acceptance_rate/(self.len_metropolis-1))*100
        print("The acceptance rate was : ", acceptance_rate, '%')
        
    def compute_parameter_estimation(self, burnin = 10):
        """ Compute an estimation of each parameter by taking the average of the corresponding Metropolis subchain after removing the first Metropolis simulations (burnin)"""
        u_heat_est = np.mean(self.u_heat_evolution[burnin:])
        sigma2_est = np.mean(self.sigma2_evolution[burnin:])
        sigma2_s_param_est = np.mean(self.sigma2_s_param_evolution[burnin:])
        sigma2_g_param_est = np.mean(self.sigma2_g_param_evolution[burnin:])
        print("u_heat = ", u_heat_est)
        print("sigma2 = ", sigma2_est)
        print("sigma2_s_param = ", sigma2_s_param_est)
        print("sigma2_g_param = ", sigma2_g_param_est)
        return(u_heat_est, sigma2_est, sigma2_s_param_est, sigma2_g_param_est)
    
    def plot_parameter_evolution(self, burnin = 0):
        x = np.arange(self.len_metropolis - burnin)
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.plot(x, self.u_heat_evolution[burnin:])
        ax1.set_xlabel("Metropolis steps")
        ax1.set_ylabel("u_heat")
        #ax1.set_title("PMCMC results : u_heat")
        
        ax2 = fig.add_subplot(222)
        ax2.plot(x, self.sigma2_evolution[burnin:], color = "limegreen")
        ax2.set_xlabel("Metropolis steps")
        ax2.set_ylabel("sigma2")
        #ax2.set_title("PMCMC results : sigma2")
        
        ax3 = fig.add_subplot(223)
        ax3.plot(x, self.sigma2_s_param_evolution[burnin:], color = "royalblue")
        ax3.set_xlabel("Metropolis steps")
        ax3.set_ylabel("sigma2_s_param")
        #ax3.set_title("PMCMC results : sigma2_s_param")
        
        ax4 = fig.add_subplot(224)
        ax4.plot(x, self.sigma2_g_param_evolution[burnin:], color = "orange")
        ax4.set_xlabel("Metropolis steps")
        ax4.set_ylabel("sigma2_g_param")
        #ax4.set_title("PMCMC results : sigma2_g_param")
        
        fig.tight_layout()
        plt.show()
    
    def plot_acf_PMMH(self, burnin):
        from statsmodels.graphics.tsaplots import plot_acf
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        plot_acf(self.u_heat_evolution[burnin:], ax = ax1, color = 'tomato')
        ax1.set_title("ACF u_heat")
        
        ax2 = fig.add_subplot(222)
        plot_acf(self.sigma2_evolution[burnin:], ax = ax2, color = "limegreen")
        ax2.set_title("ACF sigma2")
        
        ax3 = fig.add_subplot(223)
        plot_acf(self.sigma2_s_param_evolution[burnin:], ax = ax3, color = "royalblue")
        ax3.set_title("ACF sigma2_s_param")
        
        ax4 = fig.add_subplot(224)
        plot_acf(self.sigma2_g_param_evolution[burnin:], ax = ax4, color = "orange")
        ax4.set_title("ACF sigma2_g_param")
        
        fig.tight_layout()
        plt.show()

