'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/Gray-Level_Quantization.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Tuesday, February 16th 2021, 6:22:39 pm
Author: Athansya

Copyright (c) 2021 Your Company
Thresholding
Is an image processing technique that consists of assigning either the value 1 or 255 depending a given threshold.
Otsu's Thresholding method is an adaptive way of assigning the finding the optimal threshold value to divide classes.
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

# Global Thresholding
global_threshold = np.copy(img)
global_threshold[global_threshold < np.floor(256/2)] = 0
global_threshold[global_threshold > np.floor(256/2)] = 255

# Otsu's Thresholding

# First let's obtain the histogram
hist = cv2.calcHist([img],[0],None,[256],[0,256]).astype(int)
plt.plot(hist)

# Normalized histogram
# Technique consisting into transforming the discrete distribution of intensities into a discrete distribution of probabilities. To do so, we need to divide each value of the histogram by the number of pixels in the image.
P_i = hist/(img.shape[0]*img.shape[1])
plt.plot(hist_norm)

# Set threshold
k = np.arange(1, 255)

# Average whole image intensities
m_g = 0
for m in np.arange(255):
    m_g += m*P_i[m]
    
# Variables to evaluate threshold k
n = []
o_B = np.array([]) # Between-class variance

o_G = 0 # Global variance
for n in np.arange(255):
    o_G += (n - m_g)**2 * P_i[n]


for i in k:
    # Probability of class c_1 and c_2
    P_1 = 0
    P_2 = 0
    # Mean intensity value of pixels in c_1 and c_2
    m_1 = 0
    m_2 = 0
    # Cumulative mean
    m_k = 0
    for j in np.arange(i):
        P_1 += P_i[j]
        # 2)
        m_1 += j*P_i[j]
        # 3)
        m_k += j*P_i[j]
        
    for l in np.arange(i+1,255):
        P_2 += P_i[l]
        # 2)
        m_2 += l*P_i[l]
    
    res = (m_g * P_1 - m_k)**2 / (P_1*(1-P_1))
    o_B = np.append(o_B,res)
    
n = o_B/o_G

# Find optimal threshold value
optimal_value = np.where(n == np.max(n))

# Resulting image
Otsus_threshold = np.copy(img)
Otsus_threshold[Otsus_threshold <= optimal_value] = 0
Otsus_threshold[Otsus_threshold > optimal_value] = 255


# OpenCV Otsu's threshold implementation
cv2_otsus = cv2.threshold(img, 0,255, type=cv2.THRESH_OTSU)

# Display results
disp_img([img, Otsus_threshold, cv2_otsus[1]],['Original',"Otsu's threshold implementation", "OpenCV Otsu's threshold"],(15,10),1,3)
# Pretty similar, right?