"""
Created on Fri May 15 21:33:57 2020

@author: wei
"""

import numpy as np
import math

'''
levy distribution
'''

def levy_flight(n, dim, Lambda=1.5):
    sigma = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                      / (Lambda * math.gamma((1 + Lambda) / 2)) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
    u = np.random.randn(n, dim) * sigma
    v = np.random.randn(n, dim)
    step = u / np.power(np.fabs(v), 1 / Lambda)
    return step    


'''
function
'''
def ackley(x, a=20, b=0.2, c=2*np.pi):
    '''
    x = np.array((your N, your dim))  #input is matrix of n*d
        or 
        list[x1, x2, x3, x4,......,xd] 
    
    output = np.array((your N, fitness))   #it is matrix of n*1. 
    '''
    if type(x) != np.ndarray:
        x = np.array(x).reshape(1,-1)
    d = x.shape[1]
    sum1 = (-b) * ((np.sum(x ** 2, axis = 1).reshape(len(x),-1)/d) ** 0.5)
    sum2 = np.sum(np.cos(c * x), axis = 1).reshape(len(x),-1)/d
    y = (-a) * np.exp(sum1) - np.exp(sum2) + a + np.exp(1)
    return y