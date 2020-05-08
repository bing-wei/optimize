# -*- coding: utf-8 -*-
"""
firefly algorithm
"""

import numpy as np
from time import time
import matplotlib.pyplot as plt

from ackley import ackley
#%%
class FA():
    def __init__(self, bounds, popN, dim, fun=ackley, its=1000):
        self.upbound = max(bounds)
        self.lowbound = min(bounds)
        self.diff = abs(self.upbound - self.lowbound) #Full distance

        self.its = its
        self.pN = popN #Number of use points
        self.dim = dim #Dimension
        self.fun = fun
        self.call_num = 0
        self.break_num = 100000

        self.alpha = 0.2 #rand
        self.beta = 1  #attraction
        self.gamma = 0.97 #variation of the attractiveness

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
        
    def iterator(self):
        for i in range(self.pN):
            for j in range(self.pN):
                if self.fitness[i] > self.fitness[j]:
                    rij = np.linalg.norm(self.pop[i] - self.pop[j])

                    self.pop[i] = self.exceed_bound(self.pop[i] + \
                             self.beta*np.exp(-self.gamma*(rij**2)) * (self.pop[j]- self.pop[i]) + \
                             self.alpha*(np.random.uniform(0, 1)-0.5) * 40) 
                             
                    self.fitness[i] = self.function(self.pop[i].reshape(1,-1))
                if self.call_num >= self.break_num:
                    break
            if self.call_num >= self.break_num:
                break
            
        self.history.append(self.fitness[np.argmin(self.fitness)].item())    
        self.global_best = self.pop[np.argmin(self.fitness)]
    
        
    def exceed_bound(self,pop):
        new_pop = np.random.rand(1, self.dim) * self.diff + self.lowbound
        pop_bool = ((pop < self.lowbound)|(pop > self.upbound))
        pop = np.where(pop_bool, new_pop, pop)
        return pop
        
    def fit(self):
        self.init_Population()
        for its in range(self.its):
            self.iterator()
            self.pop_history.append(self.pop)
            if self.call_num >= self.break_num :
                break
        return self.global_best, self.fitness[np.argmin(self.fitness)], self.history
                

#%%
if __name__ == '__main__':
    start = time()
    function = ackley
    bounds = (40,-40)
    dim = 10
    pop = 40
    its = 10000
    
    FA_opt = FA(bounds, pop, dim, function, its=its)
    FA_opt.break_num = 10 ** 5
    FA_opt.grmma = 0.01
    FA_opt.alpha = 1
    bestpoint, bestfitness, his = FA_opt.fit()
    end = time()
    print('total time:%4.2fs'%(end-start))
    print('FA call ackley function:%6d'%FA_opt.call_num)
    print('best fitness:%2.4f'%bestfitness)
    plt.figure()
    plt.title('FA')
    plt.plot(range(len(his)),his,label='FA')


