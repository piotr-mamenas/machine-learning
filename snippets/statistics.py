import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from pylab import *
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import poisson

#Median, Avg, Std, Variation

incomes = np.random.normal(27000, 15000, 500)
plt.hist(incomes, 50)
plt.show()

median_incomes = np.median(incomes)

incomes = np.append(incomes, [100000000000])
median_incomes = np.median(incomes)

ages = np.random.randint(18, high=90, size=500)

incomes = np.random.normal(5000,3000, 10000)

plt.hist(incomes, 100)
plt.show()

print(incomes.std())
print(incomes.var())

# 34,1% - (-1,1)
# 13.6% - (-2,2)
# < 2.1%  - (-3,3)

values = np.random.uniform(-10.0, 10.0, 100000)
plt.hist(values, 50)
plt.show()

x = np.arange(-3,3,0.001)
plt.plot(x, norm.pdf(x))
plt.show()

n, p = 10, 0.5
x = np.arange(0, 10, 0.001)
plt.plot(x, binom.pmf(x, n, p))
plt.show()

# Poisson, A shop has on average 400 visitors a day, whats the probability a shop has exactly
# example 420 visitors on a given day

mu = 400
x = np.arange(300,500,0.5)

plt.plot(x, poisson.pmf(x,mu))
plt.show()

# Percentile

vals = np.random.normal(0, 0.5, 10000)

plt.hist(vals, 50)
plt.show()

print(np.percentile(vals, 50))
print(np.percentile(vals,90))

# Moment
# 1 - Avg
# 2 - Var
# 3 - Skew
# v < 0 => left-skew, v < 0 => right-skew
# 4 - Kurtosis
# How the data concentrates on a given point, higher spire => higher difference

vals = np.random.normal(0, 0.5, 10000)

plt.hist(vals, 50)
plt.show()

print(sp.skew(vals))
print(sp.kurtosis(vals))

# Correlation
# -1 perfect inverse, 0 - none, 1 perfect

def diff_mean(x):
    xmean = mean(x)
    return [xi - xmean for xi in x]

def covariance(x,y):
    n = len(x)
    return dot(diff_mean(x), diff_mean(y)) / (n-1)

def correlation(x,y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x,y) / stddevx / stddevy

pageLoads = np.random.normal(100, 10, 100)
purchaseTotalAmount = np.random.normal(1000, 30, 100)

scatter(pageLoads, purchaseTotalAmount)

print(covariance(pageLoads, purchaseTotalAmount))

print(correlation(pageLoads,purchaseTotalAmount))

# Exists a numpy func but incor result d t rounding
display(np.corrcoef(pageLoads, purchaseTotalAmount))

# Conditional probability
