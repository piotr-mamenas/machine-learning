import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier

mnist = fetch_mldata('MNIST original')

x, y = mnist["data"], mnist["target"]
print(x.shape)
print(y.shape)

some_digit = x[33456]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

print(x[:60000])

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5) # True for all 5's, False for all other digits
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

# guess if number under 33456 is a 5
lu1 = sgd_clf.predict([some_digit])
print(lu1)