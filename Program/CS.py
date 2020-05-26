# -*- coding: utf-8 -*-
"""
levy filghts with cuckoo search 
"""
from time import time 
import numpy as np
import matplotlib.pyplot as plt

from util import ackley, levy_flight

#%%
class CS():
    def __init__(self, bounds, popN, dim, its=1000, fun=ackley):
        '''
        parameter of cuckoo search
        Lambda:
            About levy_flight's lambda. Usually set to 1.5.
        pa:
            switching probability. you can set (0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5).
            But 0.25 may be a good select.
        alpha:
            About levy_flight's step size. Usually set to 0.01.
        '''
        self.Lambda = 1.5
        self.pa = 0.25
        self.alpha = 0.01
    
        #set bounds
        self.upbound = max(bounds)
        self.lowbound = min(bounds)
        self.diff = abs(self.upbound - self.lowbound) #Full distance

        self.its = its
        self.pN = popN #Number of use points
        self.dim = dim #Dimension
        self.fun = fun


    def function(self, x):
        self.call_num += x.shape[0]
        return self.fun(x)
    
    def init_Population(self):
        self.pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        self.call_num = 0
        self.fitness = self.function(self.pop)
        self.arg = np.argmin(self.fitness)
        self.best_fitness = self.fitness[np.argmin(self.fitness)]
        self.best_pop = self.pop[np.argmin(self.fitness)]
        self.history = [self.best_fitness.item()]
        self.pop_history = [self.pop]
        
        
    def iterator(self):
        #levy flight
        self.pop = self.exceed_bound(self.pop + self.alpha * levy_flight(self.pN, self.dim, Lambda=self.Lambda))
        self.fitness = self.function(self.pop)
        
        #Randomly occupy the nest
        j = np.random.randint(0, self.pN, size=(self.pN))
        c = self.fitness > self.fitness[j]
        self.fitness = np.where(c, self.fitness[j], self.fitness)
        self.pop = np.where(c, self.pop[j], self.pop)
        
        #Kick out of the nest
        c2 = np.random.rand(self.pN,1) < self.pa
        npop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        self.pop = np.where(c2, npop, self.pop)
        
        #chenck min
        if self.best_fitness > self.fitness[np.argmin(self.fitness)]:
            self.arg = np.argmin(self.fitness)
            self.best_fitness = self.fitness[np.argmin(self.fitness)]
            self.best_pop = self.pop[np.argmin(self.fitness)]
        else:
            self.fitness[self.arg] = self.best_fitness
            self.pop[self.arg] = self.best_pop
            
        #save history
        self.history.append(np.min(self.fitness))
        self.pop_history.append(self.pop)

    def exceed_bound(self,pop):
        new_pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        pop_bool = ((pop < self.lowbound)|(pop > self.upbound))
        pop = np.where(pop_bool, new_pop, pop)
        return pop
    
    def fit(self):
        self.init_Population()
        for its in range(self.its):
            self.iterator()
        return self.pop[np.argmin(self.fitness)], np.min(self.fitness), self.history
        
#%%
if __name__ == '__main__':
    start = time()
    bounds = (40,-40)
    population = 40
    Dimension = 2
    CS_opt = CS(bounds, population, Dimension, its=2499)
    bestpoint, bestfitness, his = CS_opt.fit()
    end = time()
    print('total time:%4.2fs'%(end-start))
    print('CS call ackley function:%6d'%CS_opt.call_num)
    print('best fitness:%2.4f'%bestfitness)
    plt.plot(range(len(his)),his)
    plt.show()