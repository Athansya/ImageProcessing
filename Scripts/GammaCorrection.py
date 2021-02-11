'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/GammaCorrection.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Thursday, February 4th 2021, 10:11:01 am
Author: Athansya

Copyright (c) 2021 Your Company

Description:
Power-law (gamma) transformations have the form: s=cr^gamma. Where c an gamma are positive constants. As with log transformations, 
power-law curves with fractional values of gamma map a narrow range of dark input values into a wider range of output values, with the 
opposite being true for higher values of input levels. Curves generated with values of gamma > 1 have the opposite effect as those 
generated with values of gamma < 1. When c = gamma = 1 reduces to the identity transformation.
'''
# Import packages
import matplotlib.pyplot as plt
import numpy as np

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)

# !!!! Should convert RGB to BW

# Apply gamma correction
npimg = np.asarray(img) / 255 # Must normalize to [0, 1.0] first
darker = npimg ** (1/0.5) # Darker image
brighter = npimg ** (1/1.5) # Brighter image

# Function to show Images
def disp_img(images, titles, fig_size, rows, cols):
    """[summary]
        Allows to display multiple images along with their title. Doesn't contain axis labels and the default cmap = 'gray'.
    Args:
        images ([array]): array containing images to be shown.
        titles ([array]): array containing titles to be shown.
        fig_size ([tuple]): tuple describing figure size, as in (15,10).
        rows ([type]): number of rows of the figure.
        cols ([type]): number of colums of the figure.
    """    
    fig = plt.figure(figsize = fig_size)
    for position in range(len(images)):
        fig.add_subplot(rows,cols,position+1)
        plt.imshow(images[position], cmap='gray')
        plt.axis('off')
        plt.title(titles[position])

# Plot all of them
disp_img([npimg, darker, brighter],['Original','\u03BB: 0.5','\u03BB: 1.5'], (15,5), 1, 3)


