#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
- depiction of how xor works
- just for the blog post figures

@author: nasir
"""

import numpy as np
import matplotlib.pyplot as plt

# inputs and labels
x = np.array([[0, 0], [0, 1], [1,0], [1, 1]], np.int32)
y = np.array([0, 1, 1, 0], np.int32)

# coordinates of the lines
plot_x = np.array([-0.2, 1.2])
plot_y = np.array([0.53509444, -0.88750267]) # from the xor network output
plot_y2 = np.array([1.63511384, 0.22815724])

# fill_between: fills the area between y1 and y2 for the specified x-values. default y2=0
#               x, y1, y2 all same size vector/matrix (y1 and y2 might be scalar).

# h1 unit
plt.fill_between(plot_x, y1=1.25, y2=plot_y, facecolor='lightgray', alpha=0.25) # points will be on top
plt.scatter(x[:, 0], x[:, 1], c=y, s=100)
plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25]);
plt.xticks([0.0, 0.5, 1.0]); plt.yticks([0.0, 0.5, 1.0])
plt.plot(plot_x, plot_y, color='red', linewidth=2)
plt.show()


# h2 unit
plt.fill_between(plot_x, y1=plot_y2, y2=-0.2, facecolor='lightgray', alpha=0.25)
plt.scatter(x[:, 0], x[:, 1], c=y, s=100)
plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25]);
plt.xticks([0.0, 0.5, 1.0]); plt.yticks([0.0, 0.5, 1.0])
plt.plot(plot_x, plot_y2, color='blue', linewidth=2)
plt.show()


# output units. combination of the above two
plt.fill_between(plot_x, y1=plot_y2, y2=plot_y, facecolor='lightgray', alpha=0.25)
plt.scatter(x[:, 0], x[:, 1], c=y, s=100)
plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25]);
plt.xticks([0.0, 0.5, 1.0]); plt.yticks([0.0, 0.5, 1.0])
plt.plot(plot_x, plot_y2, color='blue', linewidth=2)
plt.plot(plot_x, plot_y, color='red', linewidth=2)
plt.show()







