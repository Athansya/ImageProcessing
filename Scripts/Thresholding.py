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