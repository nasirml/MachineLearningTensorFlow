#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 - MLP for classifying the MNSIT dataset
 -

@author: nasir
"""

#%% includes

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#%% data

# read mnist data. If the data is there, it will not download again.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/nasir/dataRsrch/imageDBMain/MNIST/', one_hot=True)

'''
- Lets take a look at the data. images are 28x28 size and randomized.
- Training data consists of 55,000 samples hence the size is 55000x784. Training lebels are 55000x10
  size instead of 55000x1. This is because it in the format of one-hot vector.
- see some random images. You might have encountered this in many places alrady.
  Its kind of like a tradition to me. It has nothing to do with soul :-) you can also see corresponding
  lebels if you want.
-
'''
rand_img = np.array([56, 1001, 1025, 500])
for i in range(np.size(rand_img, 0)):
    plt.subplot(1, 4, i+1); plt.axis('off')
    plt.imshow(np.reshape(mnist.train.images[rand_img[i]], [28, 28]), cmap='gray')

print('Training data size : ', mnist.train.images.shape)
print('Training label size: ', mnist.train.labels.shape)  # labels are in one-hot vector
#print(mnist.train.labels[rand_img])


#%% create the MLP model

def multi_layer_perceptron_mnist(x, weights, biases):
    hidden_layer1 = tf.add(tf.matmul(x, weights['w_h1']), biases['b_h1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)   # apply ReLU non-linearity
    hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['w_h2']), biases['b_h2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)

    out_layer = tf.add(tf.matmul(hidden_layer2, weights['w_out']), biases['b_out'])  # NO non-linearity in the output layer

    return out_layer

#%% construct the mlp model

# hyper-parameters
learning_rate = 0.01
num_epochs = 20
batch_size = 100
display_step = 10       # display the avg cost after this number of epochs

# variables
num_input = 784         # units in the input layer 28x28 images
num_hidden1 = 128       # units in the first hidden layer
num_hidden2 = 256
num_output = 10         # units in the output layer 0 to 9. OR nClasses

# trainum_inputg data and labels
x = tf.placeholder('float', [None, num_input])     # training data
y = tf.placeholder('float', [None, num_output])    # labels

# weights and biases
weights = {
    'w_h1' : tf.Variable(tf.random_normal([num_input, num_hidden1])),       # w1, from input layer to hidden layer 1
    'w_h2' : tf.Variable(tf.random_normal([num_hidden1, num_hidden2])),     # w2, from hidden layer 1 to hidden layer 2
    'w_out': tf.Variable(tf.random_normal([num_hidden2, num_output]))       # w3, from hidden layer 2 to output layer
}
biases = {
    'b_h1' : tf.Variable(tf.random_normal([num_hidden1])),                  # b1, to hidden layer 1 units
    'b_h2' : tf.Variable(tf.random_normal([num_hidden2])),
    'b_out': tf.Variable(tf.random_normal([num_output]))
}

# construct the model
model = multi_layer_perceptron_mnist(x, weights, biases)

# cost function and optimization
loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_func)

#%% Train and test

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Train the model
for epoch in range(num_epochs):
    avg_cose = 0.0
    num_batch = int(mnist.train.num_examples / batch_size)   # total number of batches
    for nB in range(num_batch):
        trainData, trainLabels = mnist.train.next_batch(batch_size=batch_size)
        tmpCost, _ = sess.run([loss_func, optimizer], feed_dict={x: trainData, y: trainLabels})

        avg_cose = avg_cose + tmpCost / num_batch

    if epoch % display_step == 0:
        print('Epoch: %04d' %(epoch+1), 'Cost= ' + "{:.5f}".format(avg_cose))

print('Optimization done...')

# Test the model
correct_pred = tf.equal(tf.arg_max(model, 1), tf.arg_max(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
acc = accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y: mnist.test.labels})

print('Accuracy: ' + str(acc) )

sess.close()

#%%





