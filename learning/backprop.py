import numpy as np
import matplotlib.pyplot as plt

class BackProp(object):
    def forward(x, weight_left, bias_left, weight_right, bias_right):
        Z = 1 / (1 + np.exp(-x.dot(-x.dot(weight_left) - bias_left)))
        A = Z.dot(weight_right) + bias_right
        expA = np.exp(A);
        Y = expA / expA.sum(axis=1, keepdims=True)
        return Y, Z

    def classification_rate(Y, P):
        n_correct = 0
        n_total = 0
        for cnt in range(len(Y)):
            n_total += 1
            if Y[cnt] == P[cnt]:
                n_correct += 1
        return float(n_correct) / n_total

if __name__ == '__main__':
    set_size = 500
    dimension = 2
    hidden_layer_size = 3
    classes = 3

    gauss_cloud_1 = np.random.randn(set_size, dimension) + np.array([0,-2])
    gauss_cloud_2 = np.random.randn(set_size, dimension) + np.array([2,2])
    gauss_cloud_3 = np.random.randn(set_size, dimension) + np.array([-2,2])
    training_set = np.vstack([gauss_cloud_1,gauss_cloud_2,gauss_cloud_3])

    Y = np.array([0]*set_size + [1]*set_size + [2]*set_size)

    target = np.zeros((len(Y),classes))

    for cnt in range(len(Y)):
        target[cnt, Y[cnt]] = 1

    plt.scatter(training_set[:,0], training_set[:,1], c=Y, s=100, alpha=0.5)
    plt.show()
