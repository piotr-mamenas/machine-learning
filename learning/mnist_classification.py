import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

x, y = mnist["data"], mnist["target"]
print(x.shape)
print(y.shape)

some_digit = x[36000]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()