import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from loader import Loader as ld
from util import TfUtils as util

class HiddenLayer(object):
    def __init__(self, x, y, an_id):
        self.id = an_id
        self.x = x
        self.y = y
        weights, biases = util.init_weight_and_biases(x,y)
        self.weights = tf.Variable(weights.astype(np.float32))
        self.biases = tf.Variable(biases.astype(np.float32))
        self.params = [self.weights, self.biases]

    def forward(self, x):
        return tf.nn.relu(tf.matmul(x, self.weights) + self.biases)

class ANN(object):
    def __init__(self, hidden_layers_size):
        self.hidden_layers = hidden_layers_size

    def fit(self, x, y, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=400, batch_sz=100, show_fig=False):
        k = len(set(Y))

        x, y = shuffle(x,y)
        x = x.astype(np.float32)
        y = util.targets_to_indicator(y).astype(np.float32)
        x_test, y_test = x[-1000:], y[-1000:]
        y_test_flat = np.argmax(y_test, axis=1)
        x, y = x[:-1000], y[:-1000]

        n, d = x.shape
        self.hidden_layers = []
        m1 = d
        cnt = 0
        for m2 in self.hidden_layers_size:
            h = HiddenLayer(m1,m2,cnt)
            self.hidden_layers.append(h)
            m1 = m2m2
            cnt += 1
        weights, biases = util.init_weight_and_biases(m1,k)
        self.weights = tf.Variable(weights.astype(np.float32))
        self.biases = tf.Variable(biases.astype(np.float32))

        self.params = [self.weights, self.biases]
        for h in self.hidden_layers:
            self.params += h.params

        tfX = tf.placeholder(tf.float32, shape=(None, d), name='x')
        tfT = tf.placeholder(tf.float32, shape=(None, k), name='t')

        act = self.forward(tfX)
        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(act, tfT)) + rcost
        prediction = self.predict(tfX)
        train_op = tf.trainRMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)

        n_batches = n / batch_sz
        costs = []

        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                x, y = shuffle(x,y)
                for j in range(n_batches):
                    xbatch = x[j*batch_sz:(j*batch_sz + batch_sz)]
                    ybatch = y[j*batch_sz:(j*batch_sz + batch_sz)]

                    session.run(train_op, feed_dict={tfX: xbatch, tfT: ybatch})

                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={tfX:x_test, tfT: y_test})
                        costs.append(c)

                        p = session.run(prediction, feed_dict={tfX:x_test, tfT: y_test})
                        e = util.error_rate(y_test_flat, p)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, x):
        z = x
        for i in self.hidden_layers:
            z = h.forward(z)
        return tf.matmul(z, self.weights) + self.biases

    def predict(self, x):
        act = self.forward(x)
        return tf.argmax(act,1)


if __name__ == '__main__':
    X, Y = ld.get_image_data();

    model = ANN([2000,1000,500])
    model.fit(X,Y,show_fig=True)
