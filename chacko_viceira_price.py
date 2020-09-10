# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:42:56 2020

@author: Sebastian

Rough prices and volatilites according to Chacko Viceira model.

"""

import numpy as np
from fbm import FBM


#  (used for pricing in model_prices)
v_0=0.04; s_0=100; x_0=np.log(v_0); kappa=0.5; theta=x_0; mu=0.1; eta=0.001; 
 

    
# Define the sampling class to generate fBM and prices & volatility according to Heston model
class model_prices():
    """
    Euler-Maruyama scheme for computing prices and volatilities according to Chacko-Viceira model.
    
    Parameters
    ----------
    n : Integer, number of time steps.
    T : Time horizon (n/365)
    H : Hurst parameter in (0,1).
    num_sub : Number of equidistant points between trading dates for Euler Mayorama computations.
    train_samples : Amount of samples to create.
    
    Returns
    -------
    None.
        
    """
    def __init__(self, n, T, H, num_sub, train_samples):
        self.n=n; self.T=T; self.H=H; self.num_sub=num_sub; self.train_samples = train_samples
        self.a = (self.num_sub+1)*n     # a number of steps, note num_sub is the number of equidistant points between trading dates
        self.BMsampler = FBM(self.a, 0.5, self.T)
        self.fBMsampler = FBM(self.a, H, self.T)
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
        price_holder a (n+1,train_samples,2)-dim list with n the number of time_steps, 
        Ktrain the number of samples and 2 for price and volatility.
        
        """
        # do Euler for Heston with parameters specified at the beginning
        # sample increments 
        price_holder=[[] for i in range(self.n+1)]   # i-th element (shape=(train_samples,d)) i.e., sampled prices at time t_i 
        for j in range(self.train_samples):
            incr_1 = self.BMsampler.fbm()        #Brownian motion
            incr_2 = self.fBMsampler.fbm()     #Fractional Brownian motion 
            # define price_1 and volativity iteratively by Euler scheme
            price_1 = [s_0]
            vol = [v_0]
            X = [x_0]
            dt = self.t[1]-self.t[0]
            for k in range(self.a):
                price_1.append(price_1[-1] + mu*dt + np.sqrt(vol[-1])*incr_1[k])
                X.append(X[-1] + kappa*(theta-X[-1])*dt + eta*incr_2[k])
                vol.append(np.exp(X[-1]))
            # we reduce price and volatility to trading date informations
            price_1 = price_1[::self.num_sub+1]
            vol = vol[::self.num_sub+1]
            # add one sample to price_holder
            for i in range(self.n+1):
                price_holder[i].append([price_1[i], vol[i]])
                
        return price_holder
