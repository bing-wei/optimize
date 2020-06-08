# -*- coding: utf-8 -*-
"""
Bat Algorithms
"""
from time import time 
import numpy as np
import matplotlib.pyplot as plt

from util import ackley

#%%
class BA():
    def __init__(self, bounds, popN, dim, its=1000, fun=ackley):
        '''
        parameter of Bat Algorithms
        Qmin:
            frequency min.
        Qmax:
            frequency max.
        r:
            rate of pulse emission.
        A:
            loudness.
        '''
        self.Qmin = 0
        self.Qmax = 2
        self.r = 0.5
        self.A = 0.5

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
        #set init pop and init velocity
        self.pop = np.random.rand(self.pN, self.dim) * self.diff + self.lowbound
        self.velocity = np.zeros((self.pN, self.dim))
        
        #Calculation finness of pop
        self.call_num = 0
        self.fitness = self.function(self.pop)
        
        #save best pop and add history
        self.best_fitness = self.fitness[np.argmin(self.fitness)]
        self.best_pop = self.pop[np.argmin(self.fitness)]
        self.history = [self.best_fitness.item()]
        self.pop_history = [self.pop]

        
    def iterator(self):
        #Calculate velocity and new pop
        Q = self.Qmin + (self.Qmax - self.Qmin) * np.random.rand(self.pN, self.dim)
        self.velocity = self.velocity + (self.pop - self.best_pop) * Q
        newpop = self.exceed_bound(self.pop + self.velocity)
        
        #If U(0,1) > r then change to newpop otherwise change to the best_pop + 0.01 * N(0,1).
        #np.where(bool, True, False)
        newpop = np.where(np.random.rand(self.pN, 1) > self.r,
                          newpop,
                          self.best_pop.reshape(-1, self.dim).repeat(self.pN, axis=0) + \
                              0.01 * np.random.randn(self.pN, self.dim))
        newfitness = self.function(newpop)
        
        #If new fitness is better point and U(0,1) > A, change to new pop.
        b = (np.random.rand(self.pN,1) < self.A) & (self.fitness > newfitness)
        self.pop = np.where(b,
                            newpop,
                            self.pop)
        self.fitness = np.where(b,
                                newfitness,
                                self.fitness)
        
        #check best fitness
        if self.best_fitness > self.fitness[np.argmin(self.fitness)]:
            self.best_fitness = self.fitness[np.argmin(self.fitness)]
            self.best_pop = self.pop[np.argmin(self.fitness)]

        
        #save history
        self.history.append(self.best_fitness)
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
    
    BA_opt = BA(bounds, population, Dimension, its=2499)
    bestpoint, bestfitness, his = BA_opt.fit()
    end = time()
    print('total time:%4.2fs'%(end-start))
    print('BA call ackley function:%6d'%BA_opt.call_num)
    print('best fitness:%2.4f'%bestfitness)
    plt.plot(range(len(his)),his)
    plt.show()
