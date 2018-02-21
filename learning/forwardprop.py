# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:37:43 2018

@author: corpseRott
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from loader import Loader as ld


if __name__ == '__main__':
    process = ld.load_learn_process()
    
    nclass = 500
    
    X1 = np.random.randn(nclass, 2) + np.array([0,-2])
    X2 = np.random.randn(nclass, 2) + np.array([2,2])
    X3 = np.random.randn(nclass, 2) + np.array([-2,2])
    X = np.vstack([X1, X2, X3])
    
    Y = np.array([0]*nclass + [1]*nclass + [2]*nclass)
    
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()
    
    D = 2
    M = 3
    K = 3
    
    W1 = np.random.randn(D,M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M,K)
    b2 = np.random.randn(K)
    
    def forward(X, W1, b1, W2, b2):
        Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
        A = Z.dot(W2) + b2
        expA = np.exp(A)
        Y = expA / expA.sum(axis=1, keepdims=True)
        return Y
    
    def classification_rate(Y,P):
        n_correct = 0
        n_total = 0
        for i in range(len(Y)):
            n_total += 1
            if Y[i] == P[i]:
                n_correct += 1
        return float(n_correct) / n_total
    
    P_Y_given_X = forward(X, W1, b1, W2, b2)
    P = np.argmax(P_Y_given_X, axis=1)
    
    assert(len(P) == len(Y))
    
    print("Classification rate for randomly chosen weights:", classification_rate(Y,P));