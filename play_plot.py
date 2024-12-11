import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time 
# %matplotlib inline 


def y_scale_forward(x, min_y=0.0, mid_y=0.6, scaled_y=0.25, max_y=1.0):
    """
    data is min_y -> mid_y -> max_y
    want it to be plotted at min_y -> scaled_y -> max_y

    y = x.copy()
    y[y<0.6] = 0.25 * y[y<0.6]/0.6
    y[y>=0.6] = 0.25 + 0.75 * (y[y>=0.6] - 0.6) / 0.4
    return y
    """
    y = x.copy()
    y[y<mid_y] = min_y + (scaled_y - min_y) * (y[y<mid_y] - min_y) / (mid_y-min_y)
    y[y>=mid_y] = scaled_y + (max_y - scaled_y) * (y[y>=mid_y] - mid_y) / (max_y-mid_y)
    return y
    
def y_scale_inverse(x, min_y=0.0, mid_y=0.6, scaled_y=0.25, max_y=1.0):
    """
    data is min_y -> scaled_y -> max_y
    want it to be plotted at min_y -> mid_y -> max_y

    y = x.copy()
    y[y<0.25] = 0.6 * y[y<0.25] / 0.25
    y[y>=0.25] = 0.6 + 0.4 * (y[y>=0.25] - 0.25) / 0.75
    return y

    y = x.copy()
    y[y<scaled_y] = min_y + (mid_y - min_y) * (y[y<scaled_y] - min_y) / (scaled_y-min_y)
    y[y>=scaled_y] = mid_y + (max_y - mid_y) * (y[y>=scaled_y] - scaled_y) / (max_y-scaled_y)
    return y
    """
    return y_scale_forward(x, min_y=min_y, mid_y=scaled_y, scaled_y=mid_y, max_y=max_y)
  
# Example 1 
y = np.random.randn(50) 
y = y[(y > 0) & (y < 1)] 
y.sort() 
x = np.arange(len(y)) 
  
# plot with various axes scales 
plt.figure() 
  
# linear 
plt.plot(x, y) 
plt.plot(x,x/len(y))
plt.yscale("function", functions=(y_scale_forward, y_scale_inverse))
plt.grid(True) 
  
  
plt.show() 