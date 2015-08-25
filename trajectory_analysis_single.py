# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:19:28 2015

@author: Nicholas Mei

"""
import FAA_IO as FAA
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np

#color_dict and linear_gradient functions from: http://bsou.io/posts/color-gradients-with-python
#nice color gradient is: #4682B4 to #FFB347

def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

debug_path = u'C:/Users/Nicholas/Desktop/FlyBar Analysis/EtOH New Dose Expts Part 14/EtOH 85 - rev - 10 minutes set 1'

expt_set = FAA.load_set(debug_path)

expt_set.Test_Air.Trial_1.Raw_Walk_Table

#do median filtering to clean up noise in y position (the only one we care about for the turning around plotting)
data = expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table[expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table['tunnel'] == 2]['fly_y']


filter_size = 5
median_filtered = sig.medfilt(expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table[expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table['tunnel'] == 2]['fly_y'], filter_size)

median_filtered_y = sig.medfilt(expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table[expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table['tunnel'] == 2]['fly_y'], 5)
median_filtered_x = sig.medfilt(expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table[expt_set.Training_EtOH85.Trial_1.Raw_Walk_Table['tunnel'] == 2]['fly_x'], 5)

#plotting the x,y coordinates through time as a sanity check
#first set up linear color gradient
colors = linear_gradient('#4682B4', '#FFB347', n=len(median_filtered))
for timepoint in np.arange(len(median_filtered)):
    plt.plot(median_filtered_x[timepoint], median_filtered_y[timepoint], 'o', color=colors['hex'][timepoint])
    plt.pause(0.01)

#next let's take position deltas of y-position in time
diff = np.diff(median_filtered)

#then obtain a boolean array to find regions where difference values were negative 
#(i.e. regions where slope is negative and the fly is moving away from end-chamber will become TRUE)
bool_array = (diff < 0)

#if we then take the deltas of the boolean array we can get start and end indexs for clusters of "TRUE"
retreat_cluster_indxs = np.diff(bool_array).nonzero()[0]

#let's only consider retreats that were longer than 3 timepoints long
#To achieve this we subtract odd from even elemnts (i.e. end-start indexes)
min_cluster_size = 3

retreat_diffs = retreat_cluster_indxs[1::2]-retreat_cluster_indxs[::2]
number_of_retreats = np.sum(retreat_diffs > min_cluster_size)
tot_dur_of_retreats = np.sum(retreat_diffs[retreat_diffs > min_cluster_size])
