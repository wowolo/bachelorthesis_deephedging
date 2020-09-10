# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:45:48 2020

@author: Sebastian
"""
# Guarantee that working directory is the directory the current script `tester.py'
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 

#import modules
from plotter_bs import plotting_


num_eps = 5; L=3; num_nodes=16; num_sub=4; train_samples=3*10**5
do = plotting_(num_eps,L,num_nodes,num_sub,train_samples)