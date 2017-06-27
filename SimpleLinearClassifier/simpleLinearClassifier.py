
# coding: utf-8

# automatically inported from the jupyter. not bad!

# ## Simple linear classifier with TensorFlow
# In this this tutorial, we will learn how to read data from a CSV file in TensorFlow, plot them and classify with the SoftMax regression model. We will then evaluate the model on the test data and plot the decision boundary. If you are too busy (or find it silly) to read through this document, you can check my GitHub link to get the python code.
# <br /> <br />
# **Softmax** is a very population *linear* classifier. It is a generalization to the multiclass classification of binary logistic regression that "squashes" a $K$-dimensional vector ${\displaystyle \mathbf {z} } $ of arbitrary real values to a $K$-dimensional vector $p$ of real values in range (0, 1) that add up to 1. Where, 
# $$p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$ 
# <br/>
# This is also called the *softmax* function. <br /> <br />
# Lets say our data $X$ is $m \times n$, labels $y$ is $n \times 1$. Now the cross-entropy loss between the "true" distribution $y$ and and the estimated distribution $p$ is defined as:
# $$ L = -\sum_j y_j \log p_j $$
# <br />
# Our goal is to minimize this cross-entropy loss $L$. Softmax tries to find the distribution where all probability mass is on the correct class, i.e. $p$ = **[0, ...1, ...,0]** contains a single 1 at the $y_i$-th position. This distribution is called by a special name: *one-hot vector*.

# ------------------

# In[1]:

# includes
import numpy as np
import tensorflow as tf
import csv    
import matplotlib.pyplot as plt


# In[2]:

# Global variables
NUM_LABELS = 2         # y is 0 and 1. Binary classifier
BATCH_SIZE = 100
NUM_EPOCHS = 100


# ### Read data from the file
# Read the data from the CSV file and see how do they look like.

# In[3]:

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


# In[4]:

# Read the data file and show them
trainData, trainLabels = readDataFromCSV('simdata/linear_data_train.csv') # 1000x2, 1000x2 (one hot)
testData, testLabels = readDataFromCSV('simdata/linear_data_eval.csv')    # 

trainSize, numFeatures = np.shape(trainData)
print(trainSize, numFeatures)
print(trainLabels.shape)

# plot. c for color. since trainLabels is 1000x2, use [:, 1] instead of just trainLabels
plt.scatter(trainData[:, 0], trainData[:, 1], c=trainLabels[:, 1], edgecolor='w', linewidth=0.25) 


# ------------

# ### Train the model
# To train the model, we have to write a cost function. Cost function is sometimes also called loss function. General rule is, the model tryies to minimize the loss. Here we are using softmax with cross-entropy loss. Softmax is aready build-in in the TensorFlow which we can invoke by ```tf.nn.softmax(tf.matmul(x, W) + b)```. And then we write the cross-entropy loss function. For details on the softmax, please refer to the description above.
# <br /> <br />
# Size of $X, W, b$ and $y$ matrix. Consider, $nE$ = number of examples, $nF$ = number of features, and $nC$ = number of classes/labels. Then, 
# $$X = nE \times nF$$ $$W = nF \times nC$$ $$b = nC \times 1$$ $$y = nE \times 1$$

# In[5]:

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


# ### Plot the decision boundary
# Consider first the weights and biases. Intuitively, the classifier should find a separating hyperplane between the two classes, and it probably isn’t immediately obvious how $W$ and $b$ define that. For now, consider only the first column with $w_1=-10.72918034, w_2=-11.53842258$ and the corresponding $b=8.89335346$. Recall that $w_1$ is the parameter for the $x$ dimension and $w_2$ is for the $y$ dimension (not to confuse, $y$ is now the second coordinate, NOT the labels). The separating hyperplane satisfies $Wx + b=0$; from which we get the standard $y = mx + b$ form.
# 
# $$W\, \,x \, \,+\, \, b = 0$$
# $$w_1 \, x \,+ w_2 \, y \, \,+ \, \,b = 0$$
# $$y = (-w_1/w_2)\, \, \,x \, \,– \, \,b/w_2$$
# 
# For the parameters learned above, we have the line:
# 
# $$y = -0.9298654357309905 \, \, x \, \, + \, \, 0.7707599022604006$$
# 
# Here’s the plot with the line, showing it is an excellent fit for the training data.

# In[6]:

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


# ---------------

# ### Evaluate the model
# Model is supposed to fit in the training data as it's trained on them. How about on the data the model have not seen yet? Now lets evaluate how our softmax model is doing on the test set. 

# In[9]:

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


# In[10]:

predClass = tf.argmax(y, 1)
correctPred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, "float"))

print('Accuracy: ', accuracy.eval(session=sess, feed_dict={x: testData, y_: testLabels}))

predFunc = lambda X: predClass.eval(session=sess, feed_dict={x: X})

plotDecisionBoundary(testData, testLabels, predFunc)


# As we can see, our model generalizes on the test data very well. Congratulations! Now you can go ahead and make youself a cup of coffee (you deserve it!). Or read the discussion section below. 

# -----------------

# ### Discussion
# Our softmax regression model does a good job classifying on the test set and the accuracy is 100% (yeaayy!). But, you should know that this dataset is very simple, linearly separable, and has only two dimensions. Practical and real-world data are way more complicated, requires non-linear separability, and with more than hundred or even thousand dimensions. So they need special attention and creativity to work with. This includes, (i) doing some pre-processing on the data before feeding them to the learning algorithm, (ii) designing more sophisticated cost functions, and (iii) applying regularization to control overfitting the model on the training data (which we have not discussed here). Anyways, take home from this blog is that we have learned how to write an end-to-end TensorFlow program to train a linear classifier. Now we are all set to write more cmplex machine learning programs!

# ------------
