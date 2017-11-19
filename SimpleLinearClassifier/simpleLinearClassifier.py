#!/usr/bin/env python2
# coding: utf-8

'''
 - simple linear classification to find the separating hyperplane on simple datasets.
'''


#%%
# includes
import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

# Global variables
NUM_LABELS = 2         # y is 0 and 1. Binary classifier
BATCH_SIZE = 100
NUM_EPOCHS = 100

#%% Read the data from the CSV file and see how do they look like.

# function to extract the data
def readDataFromCSV(fileName):
    labels = []
    features = []
    with open(fileName) as csvFile:
        readCsv = csv.reader(csvFile, delimiter=',')
        for row in readCsv:
            labels.append(row[0])     # first one is the label. Format: 0,0.147562141324833,0.243518270820358
            features.append(row[1:])  # rests are the features

    features = np.matrix(features).astype(np.float32)  # convert to numpy matrix as float
    labels = np.array(labels).astype(np.uint8)         # 1000x1, convert to numpy array as int

    # return as one-hot matrix. All zeros in a row except the y_i, which is 1
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32) # 1000x2, convert to one-hot matrix. For softmax

    return features, labels

#%%

# Read the data file and show them
trainData, trainLabels = readDataFromCSV('simdata/linear_data_train.csv') # 1000x2, 1000x2 (one hot)
testData, testLabels = readDataFromCSV('simdata/linear_data_eval.csv')    #

trainSize, numFeatures = np.shape(trainData)
print(trainSize, numFeatures)
print(trainLabels.shape)

# plot. c for color. since trainLabels is 1000x2, use [:, 1] instead of just trainLabels
plt.scatter(trainData[:, 0], trainData[:, 1], c=trainLabels[:, 1], edgecolor='w', linewidth=0.25)


#%% Train the model. Do either 1 or 2.

# Train with the linear model
x = tf.placeholder("float", shape=[None, numFeatures]) # 1000x2, linear data size. Will be BATCH_SIZE x 2 here!
y_ = tf.placeholder("float", shape=[None, NUM_LABELS]) # 1000x2, labels, true distribution

W = tf.Variable(tf.zeros([numFeatures, NUM_LABELS]))        # 2x2, weights
b = tf.Variable(tf.zeros([NUM_LABELS]))                     # 2x1, biases

# 1. linear model with minimum squared error (MSE)
#y = tf.sigmoid(tf.matmul(x, W) + b)
#loss = tf.reduce_sum(tf.square(y - y_))
#optimizer = tf.train.GradientDescentOptimizer(0.01)  # learning rate = 0.01
#trainStep = optimizer.minimize(loss)

# 2. Softmax with cross-etropy loss
y = tf.nn.softmax(tf.matmul(x, W) + b)                      # 1000x2, prediction/hypothesis/estimated destribution
crossEntropy = - tf.reduce_sum(y_ * tf.log(y))
trainStep = tf.train.GradientDescentOptimizer(0.1).minimize(crossEntropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in xrange(NUM_EPOCHS * trainSize // BATCH_SIZE):
    offset = (step * BATCH_SIZE) % trainSize
    batchData = trainData[offset:(offset+BATCH_SIZE), :]    # e.g. from 100 to 200, all cols/features
    batchLabels= trainLabels[offset:(offset+BATCH_SIZE)]
    sess.run(trainStep, feed_dict={x: batchData, y_: batchLabels})

print('Weight matrix W: '); print(sess.run(W))              # see the W and b
print('Bias vector b:'); print(sess.run(b))


#%% Plot the decision boundary

# plot the decision boundary. We need only two points to draw the line

plot_x = [np.min(trainData[:, 0])-0.1, np.max(trainData[:, 1])+0.25]  # 2x1, extend the line to left and right
plot_y = -1/W[1, 0] * (W[0, 0] * plot_x + b[0]) # 2x1, Similar line can be found by using the second cols of W and b

plot_x = tf.cast(plot_x, tf.float32)            # convert to tensor

print(sess.run(plot_x))
print(sess.run(plot_y))

plt.scatter(trainData[:, 0], trainData[:, 1], c=trainLabels[:, 1], edgecolor='w', linewidth=0.25)
plt.plot(sess.run(plot_x), sess.run(plot_y), color='k', linewidth=1.5 )
plt.xlim([-0.2, 1.1]); plt.ylim([-0.4, 1.0]);
plt.title('on Training data')

#%% Evaluate on test sets.

# plotting function evaluating on the test set
def plotDecisionBoundary(X, Y, predFunc):
    # find out the plot boundary
    mins = np.amin(X, 0)
    mins = mins - 0.1 * np.abs(mins)
    maxs = np.amax(X, 0)
    maxs = maxs + 0.1 * maxs
    #print([mins, maxs])

    # generate dense grid
    xs,ys = np.meshgrid(np.linspace(mins[0,0],maxs[0,0], 300),
            np.linspace(mins[0,1], maxs[0,1], 300))
        # evaluate model on the dense grid
    Z = predFunc(np.c_[xs.flatten(), ys.flatten()]);
    Z = Z.reshape(xs.shape)

    # Plot the contour and training examples
    plt.contourf(xs, ys, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y[:,1], s= 50, edgecolor='w', linewidth=0.25)
    plt.title('on Test set')
    plt.show()


#%%

predClass = tf.argmax(y, 1)
correctPred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, "float"))

print('Accuracy: ', accuracy.eval(session=sess, feed_dict={x: testData, y_: testLabels}))

predFunc = lambda X: predClass.eval(session=sess, feed_dict={x: X})

plotDecisionBoundary(testData, testLabels, predFunc)


#%%
