# -*- coding: utf-8 -*-
"""

HW2

ACKLEY FUNCTION

Implement DE, PSO and Firefly algorithms for Ackley Function with 10 dim. and 20 dime. 

Repeat each algorithm 20 times by independently regenerate the initial points over [-32.768, 32.768]^d, d = 10 & 20.

Summarize the results based on these 20 independent replications. 
"""
#%% package
root = 'D:/course/109_2/最佳化/HW2'

import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir(root)

from PSO import PSO
from DE import DE
from FA import FA
from ackley import ackley
#%%
if __name__ == '__main__':
    function = ackley
    bounds = (40,-40)
    dim = 10
    
    DE_pop = 40
    PSO_pop = 1000
    FA_pop = 40
    
    call_function_frequency = 10 ** 5
    #set iteration number of model
    DE_its = int(call_function_frequency/DE_pop)
    PSO_its = int(call_function_frequency/PSO_pop)
    FA_call_num = int(call_function_frequency)
    
    #Differential Evolution
    DE_opt = DE(bounds, DE_pop, dim,function, its=DE_its)
    DE_bestpoint, DE_bestfitness, DE_his = DE_opt.fit()
    print('DE best fitness:%2.5f'%DE_bestfitness)
    
    #Particle Swarm Optimization
    PSO_opt = PSO(bounds, PSO_pop, dim, function, its=PSO_its)
    PSO_bestpoint, PSO_bestfitness, PSO_his = PSO_opt.fit()
    print('PSO best fitness:%2.5f'%PSO_bestfitness)
    
    #firefly algorithm
    FA_opt = FA(bounds, FA_pop, dim, function, its=FA_call_num)
    FA_opt.break_num = FA_call_num
    FA_bestpoint, FA_bestfitness, FA_his = FA_opt.fit()  
    print('FA best fitness:%2.5f'%FA_bestfitness)
    
    # plt.figure()
    # plt.plot(range(len(DE_his)),DE_his,label='DE')
    # plt.plot(range(len(PSO_his)),PSO_his,label='PSO')
    # plt.plot(range(len(FA_his)),FA_his,label='FA')
    # plt.legend()