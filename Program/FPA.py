# -*- coding: utf-8 -*-
"""
Flower Pollination Algorithms
"""

from time import time 
import numpy as np
import matplotlib.pyplot as plt

from util import ackley, levy_flight
#%%
class FPA():
    def __init__(self, bounds, popN, dim, its=1000, fun=ackley):
        '''
        parameter of Flower Pollination Algorithms
        Lambda:
            About levy_flight's lambda. Usually set to 1.5.
        pa:
            proximity probability. Initial value can set to 0.5.
            p = 0.8 may work better for most applications.
        '''
        self.Lambda = 1.5
        self.p = 0.8
    
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
        self.best_fitness = self.fitness[np.argmin(self.fitness)]
        self.best_pop = self.pop[np.argmin(self.fitness)]
        self.history = [self.best_fitness.item()]
        self.pop_history = [self.pop]
        
        
    def iterator(self):
        
        c1 = np.random.rand(self.pN,1) < self.p
        
        #True pop
        Tpop = self.exceed_bound(self.pop + 
                                 levy_flight(self.pN, self.dim) * (self.pop - self.best_pop))
        
        #False pop
        ids = np.random.randint(self.pN, size=(2, self.pN))
        Fpop = self.exceed_bound(self.pop + 
                                 np.random.rand(self.pN,1) * (self.pop[ids][0] + self.pop[ids][1]))
        
        newpop = np.where(c1, Tpop, Fpop)
        newfitness = self.function(newpop)
                
        c2 = newfitness < self.fitness
        self.pop = np.where(c2, newpop, self.pop)
        self.fitness = np.where(c2, newfitness, self.fitness)
        
        if self.best_fitness > (np.min(self.fitness)):
            self.best_fitness = self.fitness[np.argmin(self.fitness)]
            self.best_pop = self.pop[np.argmin(self.fitness)]
        
        #add new history
        self.history.append(self.fitness[np.argmin(self.fitness)].item())
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
if __name__ == '__main__' :
    start = time()
    bounds = (40,-40)
    population = 40
    Dimension = 10
    FPA_opt = FPA(bounds, population, Dimension, its=2499)
    bestpoint, bestfitness, his = FPA_opt.fit()
    end = time()
    print('total time:%4.2fs'%(end-start))
    print('FPA call ackley function:%6d'%FPA_opt.call_num)
    print('best fitness:%2.4f'%bestfitness)
    plt.plot(range(len(his)),his)
