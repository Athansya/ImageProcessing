'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/Gray-Level_Quantization.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Tuesday, February 16th 2021, 6:22:39 pm
Author: Athansya

Copyright (c) 2021 Your Company
'''
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)
img = np.asarray(img)

# Gray-scale conversion
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')
plt.axis('off')

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
        
# Gray-Level Quantization
img3 = np.floor(img / (256/3))
img5 = np.floor(img / (256/5))
img10 = np.floor(img / (256/10))

disp_img([img,img3,img5,img10],['Original','3 Levels','5 Levels','10 Levels'],(15,10),1,4)

