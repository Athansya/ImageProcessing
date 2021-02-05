'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/EdgeDetection.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Thursday, February 4th 2021, 6:16:03 pm
Author: Athansya

Copyright (c) 2021 Your Company
Description:
Edge Detection
'''

# Import packages
import numpy as np
import matplotlib.pyplot as plt

# Load image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)
npimg = np.asarray(img)
plt.imshow()

# !!!! Should convert RGB to BW

