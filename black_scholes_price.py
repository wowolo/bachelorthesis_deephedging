# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:38:59 2020

@author: Sebastian

Prices according to Black Scholes model.

"""

import numpy as np
from fbm import FBM


#  (used for pricing in model_prices)
s_0=100; sigma = 0.2 
 

    
# Define the sampling class to generate fBM and prices & volatility according to Heston model
class model_prices():
    """
    Euler-Maruyama scheme for computing prices according to Black-Scholes model.
    
    Parameters
    ----------
    n : Integer, number of time steps.
    T : Time horizon (n/365)
    num_sub : Number of equidistant points between trading dates for Euler Mayorama computations.
    train_samples : Amount of samples to create.
    
    Returns
    -------
    None.
        
    """
    def __init__(self, n, T, num_sub, train_samples):
        self.n=n; self.T=T; self.num_sub=num_sub; self.train_samples = train_samples
        self.a = (self.num_sub+1)*n     #here, a number of steps and note that num_sub is the number of equidistant points between trading dates
        self.BMsampler = FBM(self.a, 0.5, self.T)
        self.t = self.BMsampler.times()
    
    
    # num_sub determines the number of midpoints in the discretization of Heston model between trading dates
    def prices(self):
        """
        Method of class model_prices, creating the price_holder variable according to inputs in class 
        initilazation.
        v_0; s_0 & kappa; theta; mu; eta the parameters for Chacko Viceira model. Defined in 
        chacko_viceira_price.py.
        
        Returns
        -------
        price_holder a (n+1,train_samples,1)-dim list with n the number of time_steps, 
        Ktrain the number of samples and 2 for price and volatility.
        
        """
        # do Euler for Heston with parameters specified at the beginning
        # sample increments 
        price_holder=[[] for i in range(self.n+1)]   # i-th element has shape=(train_samples,d) i.e., sampled prices at time t_i 
        for j in range(self.train_samples):
            W = self.BMsampler.fbm()
            # define price_1 and volativity iteratively by Euler scheme
            S = [s_0]
            for k in range(self.a):
                S.append(s_0*np.exp(-self.t[k+1]*sigma**2/2+sigma*W[k+1]))
            # we reduce price and volatility to trading date informations
            S = S[::self.num_sub+1]
            # add one sample to price_holder
            for i in range(self.n+1):
                price_holder[i].append([S[i]])
                        
        return price_holder