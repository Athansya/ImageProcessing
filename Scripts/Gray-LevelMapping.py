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
img = np.array(img)

# Convert to Grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show image dimensions
print('Image dimensions are:', img.shape)

# Show image
plt.imshow(img, cmap='gray')
plt.axis('off')
# ! DON'T USE cv2.imshow() if you're running WSL and Vscode Interactive Window, kernel dies.



