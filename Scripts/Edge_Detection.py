'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/EdgeDetection.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Thursday, February 4th 2021, 6:16:03 pm
Author: Athansya

Copyright (c) 2021 Your Company
Description:
Edge Detection
It is a common problem in image processing. What we perceive as edges in an image are strong 
changes between neighboring intensities. Since images can be interpreted as functions, we can
find these changes by taking the derivative of an image. The derivative can be calculated by 
using finite differences, an approximation that is similar to the difference quotient.
- Maier, A., Steidl, S., Christlein, V. and Hornegger, J., 2018. Medical Imaging Systems. 1st ed. 
Springer International Publishing, p.38.

For more information: 
https://mccormickml.com/2013/02/26/image-derivative/

https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
'''

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)
npimg = np.asarray(img)
plt.imshow(npimg)

# Convert image to BW
npimg = np.dot(npimg,[0.299, 0.587, 0.114]) 
plt.imshow(npimg, cmap='gray')
plt.axis('off')
print("Las dimensiones de la imagen son", npimg.shape)

# Declare filters for edge detection(Prewitt)
f1 = np.array([[1,1,1],
               [0,0,0],
               [-1,-1,-1]]) #vertical edges

f2 = np.array([[1,0,-1],
               [1,0,-1],
               [1,0,-1]]) #horizontal edges

# Define convolution functions to perform edge detection
# Could be better imo

def conv2D(img, kernel):
    """[summary]
        Performs convolution operation between an image and kernel.
    Args:
        img (numpy array): Image array.
        kernel (numpy array): filter array.

    Returns:
        output (numpy array): processed image with kernel.
    """    
    k_h, k_w = kernel.shape
    output = np.zeros((img.shape[0] - k_h + 1, img.shape[1] - k_w + 1))   
    
    # Convolution
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            output[row, col] = np.sum(img[row:row + k_h, col:col + k_w] * kernel)
    return output

def conv2Dpad(img, kernel):
    """[summary]
        Performs convolution operation between an image and kernel, but adds the 
        necessary padding to mantain the aspect ratio.
    Args:
        img (numpy array): Image array.
        kernel (numpy array): filter array.

    Returns:
        output (numpy array): processed image with kernel.
    """    
    k_h, k_w = kernel.shape
    p_h, p_w = k_h-1, k_w-1
    
    output = np.zeros((img.shape[0] - k_h + p_h + 1, img.shape[1] - k_w + p_w + 1))
    
    # Next cycles help find and add the necessary padding to keep the aspect ratio intact.
    # More on in can be found here: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
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
    
    # Convolution
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            output[row, col] = np.sum(img[row:row + k_h, col:col + k_w] * kernel)
    return output

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

disp_img([npimg, conv2Dpad(npimg, f2), conv2Dpad(npimg, f1)], ['Original','Horizontal edges','Vertical edges'],(15,10),1,3)


