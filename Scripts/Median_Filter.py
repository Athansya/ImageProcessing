'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/EdgeDetection.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Thursday, February 4th 2021, 6:16:03 pm
Author: Athansya

Copyright (c) 2021 Your Company
Description:
The median filter works by finding the median value of the kernel.
For more information: 
Maier, A., Steidl, S., Christlein, V. and Hornegger, J., 2018. Medical Imaging Systems. 1st ed. Springer International Publishing
'''
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.util import random_noise

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)
img = np.asarray(img)

# Gray-scale conversion
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add random noise
noisy = random_noise(img, mode='s&p')

# Function to show images
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


# Median filter
def medianfilt(img, kernel_shape):
    """[summary]
        Aplies a median filter to an image, but adds the necessary padding to mantain the aspect ratio.
    Args:
        img (numpy array): Image array.
        kernel_shape (tuple): Tuple indicating kernel shape.

    Returns:
        output (numpy array): processed image with kernel.
    """
    
    k_h, k_w = kernel_shape
    p_h, p_w = k_h-1, k_w-1
    
    output = np.zeros((img.shape[0] - k_h + p_h + 1, img.shape[1] - k_w + p_w + 1))
    
    # Next cycles help find and add the necessary padding
    if k_h % 2 != 0:
        img = np.insert(img, 0, np.zeros([int(p_h/2),1]), axis=0)
        img = np.insert(img, img.shape[0], np.zeros([int(p_h/2),1]), axis=0)
    else:
        img = np.insert(img, 0, np.zeros([int(np.ceil(p_h/2)),1]), axis=0)
        img = np.insert(img, img.shape[0], np.zeros([int(np.floor(p_h/2)),1]), axis=0)
    
    if k_w % 2 != 0:
        img = np.insert(img, 0, np.zeros([int(p_w/2),1]), axis=1)
        img = np.insert(img, img.shape[1], np.zeros([int(p_w/2),1]), axis=1)
    else:
        img = np.insert(img, 0, np.zeros([int(np.ceil(p_w/2)),1]), axis=1)
        img = np.insert(img, img.shape[1], np.zeros([int(np.floor(p_w/2)),1]), axis=1)
    
    # Median filter
    for row in range(output.shape[0]):
        temp_median = np.asarray([])   
        for col in range(output.shape[1]):
            output[row, col] = np.median(img[row:row + k_h, col:col + k_w])
    return output

# Apply 3x3 median filter
median3 = medianfilt(noisy,(3,3))
# Apply 5x5 median filter
median5 = medianfilt(noisy,(5,5))

disp_img([img, noisy, median3, median5],['Original', 'Salt & Pepper Noise','3x3 Median Filter', '5x5 Median Filter'],(15,13),2,2)