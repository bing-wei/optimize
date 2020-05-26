# -*- coding: utf-8 -*-
"""
firefly algorithm
"""

import numpy as np
from time import time
import matplotlib.pyplot as plt

from util import ackley
#%%
class FA():
    def __init__(self, bounds, popN, dim, fun=ackley, its=1000):
        '''
        parameter of firefly algorithm
        inif_alpha:
            Random moving ratio. you can set to 0~1.
        beta:
            attraction. Usually set to 1.
        grmma:
            variation of the attractiveness you can set to 1000~0.0001.
        seta:
            set alpha epoch by t, but you are inif_alpha need set 1.
            sita can set None or [0.95,0.99].
        '''
        self.inif_alpha = 0.2 
        self.beta = 1  
        self.grmma = 1
        self.seta = None
        
        #set bounds
        self.upbound = max(bounds)
        self.lowbound = min(bounds)
        self.diff = abs(self.upbound - self.lowbound)
        
        self.its = its
        self.pN = popN #Number of use points
        self.dim = dim #Dimension
        self.fun = fun
        
        #if call_num == break_num, you will break the opt
        self.call_num = 0 
        self.break_num = None
        


    def function(self, x):
        self.call_num += x.shape[0]
        return self.fun(x)
    
    def init_Population(self):
        self.pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        self.fitness = self.function(self.pop)
        self.global_best = self.pop[np.argmin(self.fitness)]
        self.history = [self.fitness[np.argmin(self.fitness)].item()]
        self.pop_history = [self.pop]
        self.call_num = 0
        self.alpha = self.inif_alpha
        
    def iterator(self):
        for i in range(self.pN):
            for j in range(self.pN):
                if self.fitness[i] > self.fitness[j]:
                    rij = np.linalg.norm(self.pop[i] - self.pop[j])

                    self.pop[i] = self.exceed_bound(self.pop[i] + \
                             self.beta*np.exp(-self.grmma*(rij**2)) * (self.pop[j]- self.pop[i]) + \
                             self.alpha*(np.random.rand(1, self.dim)-0.5) * self.diff) 
                             
                    self.fitness[i] = self.function(self.pop[i].reshape(1,-1))
                if self.call_num == self.break_num:
                    break
            if self.call_num == self.break_num:
                break
        #set alpha epoch by t.
        if self.seta != None:
            self.alpha = self.alpha * self.seta
        
        self.history.append(self.fitness[np.argmin(self.fitness)].item())    
        self.global_best = self.pop[np.argmin(self.fitness)]
        self.pop_history.append(self.pop.copy())
        
    def exceed_bound(self,pop):
        '''
        if pop exceed bound for any dimension.
        set the dimension will random in bounds.
        '''
        new_pop = np.random.rand(1, self.dim) * self.diff + self.lowbound
        pop_bool = ((pop < self.lowbound)|(pop > self.upbound))
        pop = np.where(pop_bool, new_pop, pop)
        return pop
        
    def fit(self):
        self.init_Population()
        for its in range(self.its):
            self.iterator()
            if self.call_num == self.break_num :
                break
        return self.global_best, self.fitness[np.argmin(self.fitness)], self.history
                

#%%
if __name__ == '__main__':
    start = time()
    function = ackley
    bounds = (40, -40)
    dim = 2
    pop = 40

    FA_opt = FA(bounds, pop, dim, function)
    FA_opt.break_num = 10 ** 5
    bestpoint, bestfitness, his = FA_opt.fit()
    
    end = time()
    print('total time:%4.2fs'%(end-start))
    print('FA call ackley function:%6d'%FA_opt.call_num)
    print('best fitness:%2.4f'%bestfitness)
    
    plt.plot(range(len(his)), his)
