# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def get_weight_and_bias(input_size, output_size):
    W = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
    b = np.zeros(output_size)
    return W.astype(np.float32), b.astype(np.float32)

class HiddenLayer(object):
    def __init__(self, input_size, output_size, an_id):
        self.id = an_id
        self.input_size = input_size
        self.output_size = output_size
        W, b = get_weight_and_bias(input_size, output_size)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN(object):
    def __init__(self, hidden_layers_number):
        self.hidden_layers_number = hidden_layers_number

    def fit(self, data, targets, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=400, batch_sz=65, show_fig=False):
        target_length = len(set(targets))

        data, targets = shuffle(data, targets)
        data = data.astype(np.float32)
        targets = self.hotcode_targets(targets).astype(np.float32)

        x_train, x_test, y_train, y_test = train_test_split(data ,targets, test_size=0.2, random_state=42)
        y_test_flat = np.argmax(y_test, axis=1)

        n, d = x_train.shape
        self.hidden_layers = []
        count = 0
        input_index = d

        for output_index in self.hidden_layers_number:
            layer = HiddenLayer(input_index, output_index, count)
            self.hidden_layers.append(layer)
            input_index = output_index
            count += 1
        W, b = get_weight_and_bias(input_index, target_length)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        self.params = [self.W, self.b]
        for layer in self.hidden_layers:
            self.params += layer.params

        tfX = tf.placeholder(tf.float32, shape=(None, d), name='tfX')
        tfY = tf.placeholder(tf.float32, shape=(None, target_length), name='tfY')
        activation = self.forward(tfX)

        reg_cost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=tfY)) + reg_cost
        prediction = self.predict(tfX)
        train_operation = tf.train.RMSPropOptimizer(learning_rate,decay=decay, momentum=mu).minimize(cost)

        n_batches = n / batch_sz
        print(n)
        costs = []
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                x_train, y_train = shuffle(x_train, y_train)
                for j in range(n_batches):
                    x_batch = x_train[j*batch_sz:(j*batch_sz + batch_sz)]
                    y_batch = y_train[j*batch_sz:(j*batch_sz + batch_sz)]

                    session.run(train_operation, feed_dict={tfX: x_batch, tfY: y_batch})

                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={tfX: x_test, tfY: y_test})
                        costs.append(c)

                        p = session.run(prediction, feed_dict={tfX: x_test, tfY: y_test})
                        e = error_rate(y_test_flat, p)

                        print('i:',i,'j:',j,'batches:',n_batches,'cost:',c,'error:',e)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, input):
        output = input
        layer = 0

        for layer in self.hidden_layers:
            output = layer.forward(output)
        return tf.matmul(output, self.W) + self.b;

    def predict(self, x):
        activation = self.forward(x)
        return tf.argmax(activation, 1)

    def hotcode_targets(self, targets):
        size_x = len(targets)
        size_y = len(set(targets))
        indicator_matrix = np.zeros((size_x, size_y))
        for cnt in range(size_x):
            indicator_matrix[cnt, targets[cnt]] = 1
        return indicator_matrix

class Playground(object):
    def main(self):
        breast_cancer_data = load_breast_cancer()
        model = ANN([2000, 1000, 500])
        model.fit(breast_cancer_data.data, breast_cancer_data.target, show_fig=True)

if __name__ == '__main__':
    playground = Playground()
    playground.main()
