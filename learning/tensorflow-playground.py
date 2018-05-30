# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle

class HiddenLayer(object):
    def __init__(self, input_size, output_size, an_id):
        self.id = an_id
        self.input_size = input_size
        self.output_size = output_size
        W, b = get_weight_and_bias(input_size, output_size)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.B]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

    def get_weight_and_bias(self, input_size, output_size):
        W = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        b = np.zeros(output_size)
        return W.astype(np.float32), b.astype(np.float32)

class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, data, targets, learning_rate=10e-7, mu=0.99, decay=0.999, reg=10e-3, epochs=400, batch_sz=100, show_fig=false):
        target_lenght = len(set(targets))

        data, targets = shuffle(data, targets)
        data = data.astype(np.float32)
        targets = hotcode_targets(targets).astype(np.float32)

    def hotcode_targets(targets):
        size_x = len(targets)
        size_y = len(set(targets))
        indicator_matrix = np.zeros((size_X, size_Y))
        for cnt in range(size_x):
            indicator_matrix[cnt, targets[cnt]] = 1
        return indicator_matrix

class Playground(object):
    def main(self):
        ANN = ANN(3)

if __name__ == '__main__':
