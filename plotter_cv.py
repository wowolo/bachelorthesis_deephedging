# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:42:55 2020

@author: Sebastian

Different plotting and evaluation options.

"""
#import os 
#import sys 
#dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)
#sys.path.append(dir_path)
import matplotlib
matplotlib.use('pdf')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from nn_model_cv import neural_network



# Define class for model plot and evaluation
class plotting_():  
    
    # want indifference price for eps->0
    def __init__(self, num_epsilons, H, L, num_nodes, num_sub, train_samples):
        """
        Compute the necessary idifference price with for eps=2^{-6},...,2^{-num_epsilons}
        using the NN_model.py module. Then evaluate their asymptotic behavior.

        Parameters
        ----------
        num_epsilons : number of epsilons (see description of class).
        H : Hurst parameter for fractional Brownian motion used in model.
        L : Number of hidden layers in neural network model.
        num_nodes : Number of nodes in neural network model.
        num_sub : Number of substeps for price computation via Euler Mayorama scheme 
                  in chacko_viceira_price.py.
        train_samples : Amount of samples we generate for neural network to train on.

        Returns
        -------
        None.

        """
        self.num_epsilons=num_epsilons; self.H=H; self.L=L; self.num_nodes=num_nodes; self.num_sub=num_sub; self.train_samples=train_samples
        self.epsilons, self.diff_ = self.computation()
        self.log_plot()
       
        
    def computation(self):
        """
        Computes the np.array of epsilons (eps=2^{-6},...,2^{-num_epsilons}) and 
        associated array of differences (p_eps-p_0) needed for asymptotic analysis

        Returns
        -------
        epsilons : np.array of epsilons of shape (num_epsilons,).
        diff_ : np.array of differences of shape (num_epsilons,).

        """
        # entropic risk for no liability
        indiff_prices = [] 
        
        with open('nn_status.txt','w+') as f_status:
            f_status.write('Start experiment:\n')
        
        epsilons = [2**-(i+5) for i in range(self.num_epsilons,0,-1)]    # epsilon=2^-(i+5) for i=1,...,num_epsilons
        for i, eps in enumerate(epsilons):
            with open('nn_status.txt', 'a+') as f_status:
                f_status.write('Computation of %d-th round, eps = %.6f\n' %(i, eps))
                
            NN_model = neural_network(eps, self.H, self.L, self.num_nodes, self.num_sub, self.train_samples)
            riskZ, risk0 = NN_model.risks()
            indiff_prices.append(riskZ-risk0)
            
            with open('results.txt', 'w+') as f_result:
                f_result.write('The indifference prices so far are: ')
                f_result.write(' '.join(map(str, indiff_prices)))
            
            
        # indiff price for 0 transactions cost
        eps = 0
        NN_model = neural_network(eps, self.H, self.L, self.num_nodes, self.num_sub, self.train_samples)
        riskZ, risk0 = NN_model.risks()
        p0 = riskZ-risk0
        epsilons = np.array(epsilons); diff_= np.array(indiff_prices)-p0
        
        with open('results.txt', 'w+') as f_result:
            f_result.write('\nThe epsilons are : [epsilons] = ')
            f_result.write(' '.join(map(str, epsilons)))
            f_result.write('\nThe indifference price for no transaction costs// for [epsilons] transaction costs: ')
            f_result.write(' // '.join(' '.join(map(str,x)) for x in ([p0], indiff_prices)))
        
        return epsilons, diff_
     
    
    def log_plot(self):
        """
        Plots differences to epsilons from self.computation() and computes the 
        resulting slope.

        Returns
        -------
        None.

        """
        x=np.log(self.epsilons); y=np.log(self.diff_)
        with open('results.txt', 'a+') as f_result:
            f_result.write('We tried to plot \nx and y:')
            f_result.write('\nx: ')
            f_result.write(' '.join(map(str,x)))
            f_result.write('\ny: ')
            f_result.write(' '.join(map(str,y)))
        
        plt.plot(x, y)
        plt.savefig('asymptotic_test')
        lin_model = LinearRegression()
        lin_model.fit(x.reshape(-1,1), y.reshape(-1,1))
        
        with open('results.txt', 'a+') as f_result:
            f_result.write('\nThe coefficient of the linear regression of the asymptotic is: ')
            f_result.write(str(lin_model.coef_))
            
        print(lin_model.coef_)