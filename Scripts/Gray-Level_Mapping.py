'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/Gray-LevelMapping.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Wednesday, February 10th 2021, 10:27:52 am
Author: Athansya

Copyright (c) 2021 Your Company
Description
Gray-Level Mapping
It is a graph that shows how a pixel value in the input image (horizontal axis) maps to a pixel vaue in the output image (vertical axis).
Contrast and brightness manipulations added for clarity
- R. Paulsen, Rasmus & B. Moeslund, Thomas (2020). Introduction to Medical Image Analysis. Springer Nature Switzerland
'''

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load image
img = plt.imread('/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg')

# Convert to numpy array for better manipulation
img = np.asarray(img)

# Convert to Grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show image dimensions
print('Image dimensions are:', img.shape)

# Show image
plt.imshow(img, cmap='gray')
plt.axis('off')
# ! DON'T USE cv2.imshow() if you're running WSL and Vscode Interactive Window, kernel dies.

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
        
# Brightness manipulation
# It's important to set intensities over 255 to 255, and below 0 to 0.
brighter = img.astype(int) + 100 # type must me change because uint8 only goes from 0 to 255 and starts over once it reaches the end. e.g. 255 + 10 = 10
brighter[brighter > 255] = 255 # Numpy indexing is faster than looping over and over

darker = img.astype(int) - 100
darker[darker < 0] = 0

disp_img([img, brighter, darker],['Original', 'Brighter', 'Darker'],(15,10),1,3)

# Contrast manipulation
# Same conditions apply here
h_contrast = np.dot((img.astype(int)-128), 50) + 128
h_contrast[h_contrast > 255] = 255 

l_contrast = np.dot((img.astype(int)-128), -50) + 128
l_contrast[l_contrast < 0] = 0

disp_img([img, h_contrast, l_contrast],['Original', 'High Contrast', 'Low Contrast'],(15,10),1,3)

