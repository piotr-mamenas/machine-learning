import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import poisson

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