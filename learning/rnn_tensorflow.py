import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from loader import Loader as ld
from util import TfUtils as util
from train_test_split import TrainTestSplit as tts
from sklearn.datasets import load_boston

class BaseRNN(object):
    def __init__(self, inputs_no, neurons_no):
        self.inputs_no = inputs_no
        self.neurons_no = neurons_no
        
    def run(self, x0_batch, x1_batch):
        x0 = tf.placeholder(tf.float32, [None, self.inputs_no])
        x1 = tf.placeholder(tf.float32, [None, self.inputs_no])
        
        wx = tf.Variable(tf.random_normal(shape=[self.inputs_no, self.neurons_no],dtype=tf.float32))
        wy = tf.Variable(tf.random_normal(shape=[self.neurons_no, self.neurons_no],dtype=tf.float32))
        
        b = tf.Variable(tf.zeros([1, self.neurons_no], dtype=tf.float32))
        
        y0 = tf.tanh(tf.matmul(x0,wx) + b )
        y1 = tf.tanh(tf.matmul(y0,wy) + tf.matmul(x1,wx) + b)
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            init.run()
            y0_val, y1_val = sess.run([y0,y1], feed_dict={x0: x0_batch, x1: x1_batch})
            print(y1_val)
        
class RNN(object):
    def __init__(self, n_steps, n_inputs, n_neurons, n_outputs, learning_rate, n_epochs, batch_size):
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        
    def run(inputs, targets, test_inputs, test_targets):
        x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        y = tf.placeholder(tf.int32, [None])
        
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)
        
        logits = tf.layers.dense(states, self.n_outputs)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits = logits)
        
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        training_op = optimizer.minimize(loss)
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            init.run()
            for epoch in range(self.n_epochs):
                for iteration in range(self.n_batches)
                    x_batch, y_batch = inputs[iteration:batch_size], targets[iteration:batch_size]
                    x_batch = x_batch.reshape((-1, self.n_steps, self.n_inputs))
                    sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
                acc_train = accuracy.eval(feed_dict={X: x_batch, y: y_batch})
                acc_test = accuracy.eval(feed_dict={X: x_test, y: y_test})
                print(epoch, "Train Accuracy:", acc_train, "Test Accuracy", acc_test)
        
if __name__ == '__main__':
    x0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
    x1_batch = np.array([[9,7,8],[7,6,5],[4,3,2],[1,0,9]])
    
    model = BaseRNN(3,5)
    model.run(x0_batch,x1_batch)

    boston = load_boston()
    print(boston.data.shape)
    print(type(boston))
    print(boston.target)
    
    x_train, x_test, y_train, y_test = tts.train_test_split(boston.data, boston.target)
    print(x_train.shape)
    n_steps, n_inputs, n_neurons, n_outputs, learning_rate, n_epochs, batch_size
    model2 = RNN(16,16,100,10,0.001, 100, 47)
    
    model.run(x_train, y_train, x_test, y_test)