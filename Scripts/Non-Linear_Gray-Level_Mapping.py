'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/GammaCorrection.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Thursday, February 4th 2021, 10:11:01 am
Author: Athansya

Copyright (c) 2021 Your Company

Description:
In some, cases it is not possible to achieve a good result using a linear mapping of gray values and it is necessary to use a non-linear mapping. Some examples are shown below.
'''
# Import packages
import matplotlib.pyplot as plt
import numpy as np

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)

# !!!! Should convert RGB to BW

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


'''
Gamma Mapping
The gamma mapping is deﬁned so that the input and output 
pixel values are in the range [0, 1]. It is therefore 
necessary to normalize pixel values. Output should be 
scaled to the normal range.
'''
img = np.asarray(img)
npimg = img / 255 # Must be normalized to [0, 1.0] first
darker = npimg ** (1/0.5) # Darker image
brighter = npimg ** (1/1.5) # Brighter image

# Figure
disp_img([npimg, darker, brighter],['Original','\u03BB: 0.5','\u03BB: 1.5'], (15,5), 1, 3)

'''
Logarithmic Mapping
Each pixel is replaced by the logarithm of the pixel value. This has the effect that low intensity pixel values are enhanced. It is often used in cases where the dynamic range of the image is to great to be displayed or in images where there are a few very bright spots on a darker background. The behavior of the logarithmic mapping can be controlled by changing the pixel values of the input image using a linear mapping before the logarithmic mapping.
''' 
c = 255 / (np.log(1 + np.max(img)))
log_mapping = c * (np.log(1 + img.astype(int)))
log_mapping[log_mapping < 0] = 0
log_mapping[log_mapping > 255] = 255
disp_img([img, log_mapping.astype(int)], ['Original','Log Mapping'],(15,10),1,2)
