# -*- coding: utf-8 -*-
"""
ACKLEY FUNCTION
"""
import numpy as np
def ackley(x, a=20, b=0.2, c=2*np.pi):
    if type(x) != np.ndarray:
        x = np.array(x).reshape(1,-1)
    d = x.shape[1]
    sum1 = (-b) * ((np.sum(x ** 2, axis = 1).reshape(len(x),-1)/d) ** 0.5)
    sum2 = np.sum(np.cos(c * x), axis = 1).reshape(len(x),-1)/d
    y = (-a) * np.exp(sum1) - np.exp(sum2) + a + np.exp(1)
    return y