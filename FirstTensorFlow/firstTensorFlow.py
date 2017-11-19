#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
     - A very first program to start with the TensorFlow
     - Examples of some preliminary operations in TensorFlow.
"""

#%% includes
import tensorflow as tf


#%%
gretings = tf.constant('Hello, World !')

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# create a session and print the values of the nodes
sess = tf.Session()
print(sess.run(gretings))
print(sess.run([node1, node2]))

# build more complicated computations by combining Tensor nodes with operations (also nodes).
node3 = tf.add(node1, node2)
print('node3 : ', node3)
print('node3 value: ' + str(sess.run(node3)))

# placeholders, external inputs. A promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adderNode = a + b                   # + provides a shortcut for tf.add(a, b)
multiplierNode = a * b
print(sess.run(adderNode, {a: 3, b: 4.5}))
print(sess.run(adderNode, {a: [1, 2], b: [3, 4]}))
print(sess.run(multiplierNode, {a: -3, b: 6}))

#%%
# variables
W = tf.Variable([5.5], tf.float32)
b = tf.Variable([-2.3], tf.float32)
x = tf.placeholder(tf.float32)
y = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, feed_dict={x: [1,2,3,4]}))


# use for loop
W = tf.Variable([5.5], tf.float32)
b = tf.Variable([-2.3], tf.float32)
x = tf.placeholder(tf.float32)
#W = tf.add(W, 0.5)
y = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
for i in range(4):
    print(sess.run(y, feed_dict={x: i+1}))

# placeholder
const1 = tf.constant(-10.0, dtype=tf.float32)
const2 = tf.constant([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]], dtype=tf.float32)
print(sess.run(const1))
print(sess.run(const2))

# another example of variable with matrix multiplication
mat_a = tf.placeholder(tf.float32, shape=[3, 2])
mat_b = tf.placeholder(tf.float32, shape=[2, 3])
mat_c = tf.matmul(mat_a, mat_b)        # result will be 3x3 matrix
mat_d = tf.matmul(mat_c, const2)       # multiply again with the constant
result = sess.run(mat_d, feed_dict={mat_a: [[1, 2], [3, 4], [5, 6]], mat_b: [[1, 2, 3], [4, 5, 6]]})
print(result)


#%%