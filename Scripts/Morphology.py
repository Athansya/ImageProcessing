'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/Morphology.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Wednesday, March 17th 2021, 11:23:15 am
Author: Athansya

Description:
Morphology operates like the other neighborhood processing methods by applying
a kernel to each pixel in the input. In morphology, the kernel is denoted a structuring
element and contains “0”s and “1”s. You can design the structuring element as you
please, but normally the pattern of “1”s form a box or a disc. Which type and size to use
is up to the designer, but in general a box-shaped structuring element tends to preserve 
sharp object corners, whereas a disc-shaped structuring element tends to round the corners
of the objects.

A structuring element is not applied in the same way as we saw in the previous
chapter for the kernels. Instead of using multiplications and additions in the calculations,
a structuring element is applied using either a Hit or a Fit operation. Applying
one of these operations to each pixel in an image is denoted Dilation and Erosion,
respectively. Combining these two methods can result in powerful image processing
tools known as Compound Operations. 

- Paulsen, R.R. & Mo T.B., 2020. Introduction to Medical Image Analysis. 1st ed. Springer 
International Publishing, p.75.
'''
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Structuring elements. Later on, I'll code them myself with a little more time.
from skimage.morphology import (square, rectangle, diamond, disk, octagon, star)

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)
img = np.asarray(img)

# Gray-scale conversion
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap = 'gray')

# ! You should check if it is normalized before applying the following cv2 code
# np.max(img)
# np.min(img)

# Create a binary mask with Otsu's thresholding method
_,mask = cv2.threshold(img, 0,255, type=cv2.THRESH_OTSU)
plt.imshow(mask, cmap='gray')

'''
Hit Operation

For each “1” in the structuring element we investigate whether the pixel at the same
position in the image is also a “1”. If this is the case for just one of the “1”s in the
structuring element, we say that the structuring element hits the image at the pixel
position in question (the one on which the structuring element is centered). This pixel
is therefore set to “1” in the output image. Otherwise it is set to “0”.

Fit Operation

For each “1” in the structuring element we investigate whether the pixel at the same
position in the image is also a “1”. If this is the case for all the “1”s in the structuring
element, we say that the structuring element ﬁts the image at the pixel position in
question (the one on which the structuring element is centered). This pixel is therefore set 
to “1” in the output image. Otherwise it is set to “0”. 
'''

# Once we defined what Hit and Fit operations are, we can continue with Dilation and Erosion.
# Applying Hit to an entire image is called Dilation. 
# Applying Fit to an entire image is called Erosion.

# We define our functions here
def dilation(img, se):
    """[summary]
        Performs dilation operation on binary image
    Args:
        img (numpy array): binary image array.
        se (numpy array): structuring element array array.

    Returns:
        output (numpy array): dilated image.
    """    
    k_h, k_w = se.shape
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
            if np.any(img[row:row + k_h, col:col + k_w]) == 1:
                output[row, col] = 1
            else:
                output[row, col] = 0
    return output

def erosion(img, se):
    """[summary]
        Performs dilation operation on binary image
    Args:
        img (numpy array): binary image array.
        se (numpy array): structuring element array array.

    Returns:
        output (numpy array): dilated image.
    """    
    k_h, k_w = se.shape
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
            if np.all(img[row:row + k_h, col:col + k_w]) == 1:
                output[row, col] = 1
            else:
                output[row, col] = 0
    return output

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

disp_img([mask, dilation(mask, square(5)), dilation(mask, square(10)), dilation(mask, square(15)),
          mask, erosion(mask, square(5)), erosion(mask, square(10)), erosion(mask, square(15))],
         ['Original', 'Dilation 5x5', 'Dilation 10x10', 'Dilation 15x15',
          'Original', 'Erosion 5x5', 'Erosion 10x10', 'Erosion 15x15'],(20,7),2,4)