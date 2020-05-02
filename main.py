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
from ackley import ackley
#%%
if __name__ == '__main__':
    function = ackley
    bounds = (40,-40)
    dim = 10
    DE_pop = 40
    DE_opt = DE(bounds,DE_pop,dim,function)
    DE_bestpoint, DE_bestfitness, DE_his = DE_opt.fit()
    plt.figure()
    plt.plot(range(len(DE_his)),DE_his)
    PSO_pop = 1000
    model = PSO(bounds, PSO_pop, DE_pop)
    PSO_bestpoint, PSO_bestfitness, PSO_his = model.fit()
    plt.plot(range(len(PSO_his)),PSO_his)
