'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/GammaCorrection.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Thursday, February 4th 2021, 10:11:01 am
Author: Athansya

Copyright (c) 2021 Your Company

Description:
Power-law (gamma) transformations have the form: s=cr^gamma. Where c an gamma are positive constants. As with log transformations, 
power-law curves with fractional values of gamma map a narrow range of dark input values into a wider range of output values, with the 
opposite being true for higher values of input levels. Curves generated with values of gamma > 1 have the opposite effect as those 
generated with values of gamma < 1. When c = gamma = 1 reduces to the identity transformation.
'''
# Import packages
import matplotlib.pyplot as plt
import numpy as np

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)
#plt.imshow(img)
#plt.axis('off')

# !!!! Should convert RGB to BW

# Apply gamma correction
npimg = np.asarray(img) / 255 # Must normalize to [0, 1.0] first
darker = npimg ** (1/0.5) # Darker image
brighter = npimg ** (1/1.5) # Brighter image

# Plot all of them. Could be made into a function
f = plt.figure(figsize=(15,5))
f.add_subplot(1,3,1)
plt.imshow(npimg)
plt.axis('off')
plt.title('Original')
f.add_subplot(1,3,2)
plt.imshow(darker)
plt.axis('off')
plt.title('\u03BB: 0.5')
f.add_subplot(1,3,3)
plt.imshow(brighter)
plt.axis('off')
plt.title('\u03BB: 1.5')

