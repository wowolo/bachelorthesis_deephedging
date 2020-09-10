# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:42:56 2020

@author: Sebastian

Neural network with architecture according to parameters (L,num_nodes), (optional) grid_search for hyperparameters,
and prediction on independent train set.

"""

import numpy as np

from tensorflow.keras.layers import Input, Dense, Concatenate, Subtract, \
                        Multiply, Lambda, Add, BatchNormalization
from tensorflow.keras import optimizers

from tensorflow.keras import Model
from tensorflow.compat.v1.keras.initializers import RandomNormal
import tensorflow.keras.backend as K

from chacko_viceira_price import model_prices
from util_cv import custom_loss, input_constructor



# Define fixed paramters for framework of every experiment:
n=30 #time-steps
d=2 #number of trading instruments - for Chacko Viceira price: d=2 and for Black Scholes price: d=1
T=float(n)/365. #finite time horizon 
wealth_0=0  #wealth at time 0
strike=100 #strike for computation of the agent's liability



# Define NN class to implement all relevant computations related to the NN.
class neural_network():
    
    def __init__(self, eps, H, L, num_nodes, num_sub, train_samples, optimizer=optimizers.Adam()):
        """
        Neural network class with build_model(), fitted_prediction() and gridcv() methods.
        
        Parameters
        ----------
        eps : Epsilons in proportional transactions costs.
        H : Hurst paramenter for fBM driving log-volatility.
        L : Indicates that we have 2L-2 hidden layers.
        num_nodes : Number of nodes of dense layers.
        num_sub : Number of sub intervals for computation of prices/volatilities (see chacko_viceira_price.py).
        train_samples : Number of samples.
        optimizer : The default is optimizers.Adam().
            
        Returns
        -------
        None.
            
        """
        self.n=n; self.d=d; self.T=T; self.wealth_0=wealth_0
        self.eps=eps; self.H=H; self.L=L; self.num_nodes=num_nodes; self.num_sub=num_sub
        self.train_samples=train_samples; self.optimizer = optimizer
    
    
    
    
    # Building of architecture  
    def build_model(self):
        """
        Method for building model according to given parameters in initialization of class.

        Returns
        -------
        Keras.Model() ready to be compiled.

        """
        w = Input(shape=(1,))
        Z = Input(shape=(1,))
        price = Input(shape=(d,))                     
        strategy = Input(shape=(d,))                # start with 0 assets at time 0
        wealth = Input(shape=(1,))
        
        inputs = [w, Z, price, strategy, wealth]
        
        # define layer structure (-> always dense layers)
        layers = []
        for j in range(self.n):
            for i in range(self.L):
                initializerk = RandomNormal(0,1e-6)
                initializerb = RandomNormal(0,0)
                if i < self.L-1:
                    nodes = self.num_nodes
                    layer_1 = Dense(nodes, activation='relu',trainable=True,
                                  kernel_initializer=initializerk,  #kernel_initializer='random_normal',
                                  bias_initializer=initializerb, 
                                  name=str(j)+str(i))
                    layer_2 = BatchNormalization()                  # Batch noramlization of the hidden layers
                    layers = layers + [layer_1] + [layer_2]
                else:
                    nodes = 2*d
                    layer = Dense(nodes, activation='linear', trainable=True,
                                  kernel_initializer=initializerk,  #kernel_initializer='random_normal',
                                  bias_initializer=initializerb,
                                  name=str(j)+str(i))
                    layers = layers + [layer]
        
        old_strategy = strategy
        for j in range(self.n):
            strategyhelper = Concatenate()([price,strategy])
            for i in range(2*self.L-1):
                strategyhelper = layers[i + j*(2*self.L-1)](strategyhelper)
            new_strategy = Lambda(lambda x: x[:,:d])(strategyhelper)
            strategyhelper = Lambda(lambda x: x[:,d:])(strategyhelper)
            incr = Input(shape=(d,), name='incr'+str(j))
            change = Subtract(name='change'+str(j))([new_strategy,old_strategy])
            absolutechanges = Lambda(lambda x: K.abs(x))(change)
            costs = Multiply()([absolutechanges,price])
            costs = Lambda(lambda x: -self.eps*K.sum(x,axis=1))(costs)          # eps as parameter (see above)
            price = Add()([incr, price])                                        # update price via increments
            mult = Multiply(name='mult'+str(j))([new_strategy, incr])               # changed: strategyhelper -> strategy
            mult = Lambda(lambda x: K.sum(x,axis=1))(mult)
            old_strategy = new_strategy
            wealth = Add()([mult,wealth])                                       # update wealth
            wealth = Add(name='wealth'+str(j))([costs,wealth])                  # update wealth
            inputs = inputs + [incr]
        #as a result we get input = [Z, price, strategy, costs, wealth, incr_1, ..., incr_n] 
     
        helper = Lambda(lambda x: K.abs(x))(new_strategy)                       # because of total liquidation at time T
        costs = Multiply()([helper,price])
        costs = Lambda(lambda x: -self.eps*K.sum(x,axis=1))(costs)        
        wealth = Add()([costs,wealth])                                          # include the transactions costs for 'complete payout' at T
        wealth = Subtract(name='wealth'+str(self.n))([Z,wealth])                # negative (sellers) wealth, Z-(\delta*S)_T+C_T
        #adapt for OCE with squared loss
        w = Dense(1, activation='linear', trainable=True,
                                  kernel_initializer=RandomNormal(0,0.001),
                                  bias_initializer=RandomNormal(0,0))(w)
        
        state = Subtract()([wealth,w])
        state = Lambda(lambda x: K.square(x))(state)
        state = Add()([state,w])
        state=wealth
        
        return Model(inputs=inputs, outputs=state)      
        
    
    
    
    
    # define the training of self.model with standard parameters or customized parameters
    def risks(self):
        """
        Method to compute the agent's risk in the market when selling the liability and when selling nothing
        using convex risk measures.
        
        Returns
        -------
        riskZ: float, "risk with liabilty Z"
        risk0: float, "risk without any sale"
        
        """
        # sample prices S_1 and S_2
        # and get into 'Input-format' for neural network
        # given: S_1 and vol list , where n+1 the number of trading dates (including 0 and T) 
        samples = model_prices(self.n, self.T, self.H, self.num_sub, self.train_samples)
        print('sampling...')
        price_holder_train = samples.prices()
        price_holder_test = samples.prices()
        
        # compute risks
        y_dummy = np.zeros((self.train_samples,1))    # dummy ytrain
        
        # compute riskZ
        bool_liab = True
        xtrain = input_constructor(price_holder_train, bool_liab) 
        print('building...[1]')
        NN_model = self.build_model()
        NN_model.compile(optimizer=self.optimizer, loss=custom_loss)
        print('fitting...[1]')
        NN_model.fit(xtrain, y_dummy, batch_size=256, epochs=75, verbose=2)
        xtest = input_constructor(price_holder_test, bool_liab)
        y = NN_model.predict(xtest)  
        #w = -np.mean(y)-0.5
        riskZ = np.mean(y)                                                     # compute with liability
        
        # compute risk0
        bool_liab = False                                                      # set liability Z to 0
        xtrain = input_constructor(price_holder_train, bool_liab) 
        print('building...[2]')                                                     
        NN_model = self.build_model()
        NN_model.compile(optimizer= self.optimizer, loss=custom_loss)
        print('fitting...[2]')
        NN_model.fit(xtrain, y_dummy, batch_size=256, epochs=75, verbose=2)
        xtest = input_constructor(price_holder_test, bool_liab)
        y = NN_model.predict(xtest) 
        #w = -np.mean(y)-0.5
        risk0 = np.mean(y)                                                     # compute risk without liability
        
        
        return riskZ, risk0
    
    
    
    
    # implement grid search of hyperparameters by cross validation
    # -> test: num_nodes, L, Adam(..), batch_size, epochs, early stopping
    
    