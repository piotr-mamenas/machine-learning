import scipy.stats as sp
import matplotlib.pyplot as plt
import numpy as np

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