#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- visualize mnist data using t-SNE
- do a round of PCA before the data sending to the t-sne. otherwise bh_tsne takes time

@author: nasir
"""

import numpy as np
from matplotlib import pyplot as plt
from tsne import bh_sne     # bh_tsne. sudo pip install tsne
#from bhtsne import tsne    # another implementation. sudo pip install bhtsne
import scipy

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/nasir/dataRsrch/imageDBMain/MNIST/', one_hot=True)

print('Training data size : ', mnist.train.images.shape)
print('Training label size: ', mnist.train.labels.shape)


# load up data
x_data = mnist.train.images[:]  # 55000x784
y_data = mnist.train.labels[:]  # 55000x10 , as one-hot vector


# For speed of computation, only run on a subset
n = 10000
x_data = x_data[:n]
y_data = y_data[:n]

# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')     # 20000x784
x_data = x_data.reshape((x_data.shape[0], -1))

# find the labels from the one-hot vector
y_data = np.squeeze(np.where(y_data == 1))[1] # gives 2x20000, seq and values. get only the values

# do PCA before t-sne
u, _, _ = scipy.sparse.linalg.svds(x_data, k=16)  # 20000x16

# perform t-SNE embedding
vis_data = bh_sne(u)       # takes a while without the PCA

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_data, edgecolors='w', linewidths=0.30) #cmap=plt.cm.get_cmap("jet", 10)
plt.xticks([]); plt.yticks([])
plt.show()