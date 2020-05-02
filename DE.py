# -*- coding: utf-8 -*-
"""
Differential Evolution
"""
#%%package
import numpy as np
from time import time
import matplotlib.pyplot as plt

from ackley import ackley

#%%Differential Evolution
class DE():
    def __init__(self, bounds, popN, dim, fun=ackley, its=1000):
        self.F = 0.8
        self.crossp = 0.5
        self.upbound = max(bounds)
        self.lowbound = min(bounds)
        self.its = its
        self.pN = popN #Number of use points
        self.dim = dim #Dimension
        self.diff = abs(self.upbound - self.lowbound) #Full distance
        self.fun = fun #your function
        
    def function(self, x):
        return self.fun(x)
    
    def init_Population(self):
        self.pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        self.fitness = self.function(self.pop)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.pop[self.best_idx]
        self.history = []
        
    def iterator(self):
        ids = np.random.randint(self.pN, size=(3, self.pN))
        donoe_vector = self.pop[ids][0] + self.F * (self.pop[ids][1] - self.pop[ids][2])
        mutant = self.exceed_bound(donoe_vector)
        cross_points = np.random.rand(self.pN, self.dim) < self.crossp
        trial_denorm = np.where(cross_points, mutant, self.pop)
        new_fitness = self.function(trial_denorm)
        change = new_fitness < self.fitness
        self.fitness = np.where(change, new_fitness, self.fitness)
        self.pop = np.where(change, trial_denorm, self.pop)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.pop[self.best_idx]
        
    def exceed_bound(self,pop):
        new_pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        pop_bool = ((pop < self.lowbound)|(pop > self.upbound))
        pop = np.where(pop_bool, new_pop, pop)
        return pop
    
    def fit(self):
        self.init_Population()
        for its in range(self.its):
            self.iterator()
            self.history.append(self.fitness[self.best_idx].item())
        return self.best, self.fitness[self.best_idx], self.history
#%%
if __name__ == '__main__' :
    start = time()
    bounds = (50,-50)
    population = 40
    Dimension = 20
    model = DE(bounds, population, Dimension)
    bestpoint, bestfitness, his = model.fit()
    end = time()
    print('total time:%4.2fs'%(end-start))
    plt.plot(range(len(his)),his)