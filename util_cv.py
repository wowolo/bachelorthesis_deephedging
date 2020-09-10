# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:43:57 2020

@author: Sebastian
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda


# Define loss for exponential utility indifference price
def custom_loss(y_true, y_pred):
    """
    Helps creating loss function for NN_model

    Parameters
    ----------
    y_true : dummy as np.array of shape (train_samples,1).
    y_pred : prediction of neural network as np.array of shape(train_samples,1).

    Returns
    -------
    Mean of y_pred as tensor.

    """
    return K.mean(y_pred)       




# Define liability Z dependent on S_1
def liability(S_1_T):
    """
    Liability with pre-defined strike (see NN_model.py).

    Parameters
    ----------
    S_1_T : Price S_1 of assets at time T.

    Returns
    -------
    np.array of shape S_1_T.

    """
    from nn_model_cv import strike
    helper = S_1_T - np.ones(np.array(S_1_T).shape)*strike
    return helper * (helper>=0)




# build inputs (i.e. xtrain)
def input_constructor(price_holder, bool_liab):
    """
    Given the prices in price_holder (see below for format), this function creates input for NN_model 
    (see NN_model.py) with or without liability depending on bool_liab.

    Parameters
    ----------
    price_holder : A 3-dim list with (n+1,train_samples,d) where n the number of time steps and d the number 
    of trading instruments i.e., price_holder[i][j][k] is at t_i of sample j the price(if k=0) / volatility(if k=1).
    bool_liab : Boolean depending whether the liability Z is 0(for bool_liab=False) or liability()
    (for bool_liab=True) with strike assigned in NN_model.py.

    Returns
    -------
    List xtrain of np.arrays for inputs of NN model (the format is the required format needed for 'inputs' 
    in NN_model.py)

    """
    from nn_model_cv import wealth_0    #get wealth at time 0 from framework in NN_model.py
    n = len(price_holder)-1          #number of time steps
    train_samples = len(price_holder[0])
    d = len(price_holder[0][0])
    
    # Define the 'components' of inputs by np.arrays
    w = [np.zeros((train_samples,1))]
    
    if bool_liab:
        Z = [liability([price_holder[-1][i][0] for i in range(train_samples)])]
    else:
        Z = [np.zeros((train_samples, 1))]    # set liability Z to 0
        
    initial_price = [np.array(price_holder[0])]
    delta_0 = [np.zeros((train_samples,d))]
    initial_wealth = [wealth_0*np.ones((train_samples,1))] 
    increments = [np.array(price_holder[i+1])-np.array(price_holder[i]) for i in range(n)]
    
    
    xtrain = (w + Z + initial_price + delta_0 + initial_wealth + increments)
        
    return xtrain