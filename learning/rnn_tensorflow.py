import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from loader import Loader as ld
from util import TfUtils as util

class RNN(object):
    def __init__(self, inputs_no, neurons_no):
        self.inputs_no = inputs_no
        self.neurons_no = neurons_no
        
    def fit(self, x0_batch, x1_batch):
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
        
        
if __name__ == '__main__':
    x0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
    x1_batch = np.array([[9,7,8],[7,6,5],[4,3,2],[1,0,9]])
    
    model = RNN(3,5)
    model.fit(x0_batch,x1_batch)