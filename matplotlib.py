import scipy.stats as sp
import matplotlib.pyplot as plt
import numpy as np
from pylab import randn

x = np.arange(-3,3,0.01)

plt.plot(x, sp.norm.pdf(x))
plt.plot(x, sp.norm.pdf(x, 1.0, 0.5))
plt.show()

# Color switching
axes = plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.xlabel('Visitors');
plt.ylabel('Probability');

plt.plot(x, sp.norm.pdf(x), 'b-')
plt.plot(x, sp.norm.pdf(x, 1.0, 0.5), 'r:')
plt.legend(['Crash during night','Crash during day'], loc=1)
plt.show()

# Pie Charting
values = [12, 55, 4, 32, 14]
colors = ['r','b','c','m','g']
explode = [0,0,0,0.3,0.4]

labels = ['Switzerland','Venezuela','Poland','USA','Rest of World']
plt.pie(values, colors=colors, labels=labels, explode=explode)
plt.title('Smiles per per day per person')
plt.show()

# Column Charting
plt.bar(range(0,5), values, color=colors)
plt.show()

# Scatter
x = randn(500)
y = randn(500)
plt.scatter(x,y)
plt.show()

# Histogram
x = np.random.normal(5000,3000,10000)
plt.hist(x,100)
plt.show()

#B&Wh Diagram, Boxplot
uniformStruct = np.random.rand(100) * 100 - 40;
high = np.random.rand(10) * 50 + 100;
low = np.random.rand(10) * -50 - 100;

data = np.concatenate((uniformStruct, high, low))
plt.boxplot(data)
plt.show()