# -*- coding: utf-8 -*-
"""
Particle Swarm Optimization
"""

import numpy as np
from time import time
import matplotlib.pyplot as plt

from ackley import ackley

class PSO():
    def __init__(self, bounds, popN, dim, fun=ackley, its=1000):
        self.alpha = 2
        self.beta = 2
        self.upbound = max(bounds)
        self.lowbound = min(bounds)
        self.its = its
        self.pN = popN #Number of use points
        self.dim = dim #Dimension
        self.diff = abs(self.upbound - self.lowbound) #Full distance
        self.fun = fun
        self.call_num = 0


    def function(self, x):
        self.call_num += x.shape[0]
        return self.fun(x)
    
    def init_Population(self):
        self.velocity = np.zeros((self.pN, self.dim))
        self.pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        self.fitness = self.function(self.pop)
        self.best = self.pop
        self.global_best = self.pop[np.argmin(self.fitness)]
        self.history = [self.fitness[np.argmin(self.fitness)]]
        self.pop_history = [self.pop]
        self.call_num = 0
        
    def iterator(self):
        #velocity vector
        velocity_1 = self.alpha * np.random.rand(self.pN, 1) * (self.global_best - self.pop) 
        velocity_2 = self.beta * np.random.rand(self.pN, 1) * (self.best - self.pop) 
        self.velocity = self.velocity + velocity_1 + velocity_2
        
        #new population
        new_pop = self.exceed_bound(self.pop + self.velocity)
        new_fitness = self.function(new_pop)
        change = new_fitness < self.fitness
        self.fitness = np.where(change, new_fitness, self.fitness)
        self.best = np.where(change, new_pop, self.pop)
        self.global_best = self.best[np.argmin(self.fitness)]
        self.pop = new_pop

        
        
    def exceed_bound(self,pop):
        new_pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        pop_bool = ((pop < self.lowbound)|(pop > self.upbound))
        pop = np.where(pop_bool, new_pop, pop)
        return pop
        
    def fit(self):
        self.init_Population()
        for its in range(self.its):
            self.iterator()
            self.history.append(self.fitness[np.argmin(self.fitness)].item())
            self.pop_history.append(self.pop)
        return self.global_best, self.fitness[np.argmin(self.fitness)], self.history
        
#%%
if __name__ == '__main__':
    start = time()
    bounds = (40,-40)
    population = 1000
    Dimension = 10
    PSO_opt = PSO(bounds, population, Dimension, its=100)
    bestpoint, bestfitness, his = PSO_opt.fit()
    end = time()
    print('total time:%4.2fs'%(end-start))
    print('PSO call ackley function:%6d'%PSO_opt.call_num)
    print('best fitness:%2.4f'%bestfitness)
    plt.plot(range(len(his)),his)

