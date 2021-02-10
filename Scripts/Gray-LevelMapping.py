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

# Brightness manipulation
# It's important to set intensities over 255 to 255, and below 0 to 0.
brighter = img.astype(int) + 100 # type must me change because uint8 only goes from 0 to 255 and starts over once it reaches the end. e.g. 255 + 10 = 10
brighter[brighter > 255] = 255

darker = img.astype(int) - 100
darker[darker < 0] = 0

fig = plt.figure(figsize=(15,10))
fig.add_subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original')
fig.add_subplot(1,3,2)
plt.imshow(brighter, cmap='gray')
plt.axis('off')
plt.title('Brighter')
fig.add_subplot(1,3,3)
plt.imshow(darker, cmap='gray')
plt.axis('off')
plt.title('Darker')
