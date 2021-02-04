'''
Filename: /home/atoriz98/PythonProjects/ImageProcessing/Scripts/histogram.py
Path: /home/atoriz98/PythonProjects/ImageProcessing/Scripts
Created Date: Wednesday, February 3rd 2021, 10:10:41 am
Author: Athansya

Copyright (c) 2021 Your Company

Description:
Histograms provide information about the distribution of the intensity values of an image and are frequently used in image segmentation and in image enhancement.
- Maier, A., Steidl, S., Christlein, V. and Hornegger, J., 2018. Medical Imaging Systems. 1st ed. Springer International Publishing, p.38.
'''

# Load packages
import numpy as np
import matplotlib.pyplot as plt

# Read and display image
path = '/home/atoriz98/PythonProjects/ImageProcessing/Scripts/Images/CovidXray.jpeg'
img = plt.imread(path)
plt.imshow(img)
plt.axis('off')

# Convert img to np array
npimage = np.asarray(img) # / 255
print(npimage)
plt.imshow(npimage)
plt.axis('off')

# Check img dimensions
print('RGB image dimensions are:',npimage.shape)

''' RGB to Grayscale
Converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components: 0.2989 * R + 0.5870 * G + 0.1140 * B 
'''
npimageG = np.dot(npimage,[0.299, 0.587, 0.114]) #dot product
print('Grayscale image dimensions are:',npimageG.shape)
plt.imshow(npimageG, cmap='gray')
plt.axis('off')

# Check max and min value
print(npimageG.shape)
print('The maximum intensity value is:{}'.format(np.max(npimage)))
print('The minimum intensity value is:{}'.format(np.min(npimage)))

# Plot histogram
bins = np.zeros(256)
for row in range(npimageG.shape[0]):
    for col in range(npimageG.shape[1]):
        value = npimageG[row, col]
        bins[int(value)] += 1 #Counts instances of the same intensity value

# Alternatively, we can use a shorter method to count instances of the same intensity value
bins_2 = np.bincount(npimageG.ravel().astype(int))
# In order to check the first implementation, we can throw in an assertion
assert np.array_equal(bins,bins_2)

# Comparsion of my histogram and Matplotlib
f = plt.figure(figsize=(15,5))
f.add_subplot(1,2,1)
plt.fill_between(range(256),bins, 0)
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 255, step=50))
plt.title('Mine')
f.add_subplot(1,2,2)
plt.hist(npimageG.ravel(), 256, [0, 256])
plt.xticks(np.arange(0, 255, step=50))
plt.title('Matplotlib')

# Note: For future comparison, timeit.timeit would be useful

'''
Cumulative distribution function (CDF)
It is a cumulative sum of all the probabilities lying in its domain and defined by:
cdf(x) = sum_{k =-\infty}^{x} P(k)
'''

# First,we need to obtain the Probability Density Function (PDF)
PDF = bins/np.sum(bins)
# Then, we compute the cumulative sum to obtain the CDF
CDF = np.cumsum(PDF)

# Finally, we plot and compare it with matplotlib implementation
f = plt.figure(figsize=(15,5))
f.add_subplot(1,2,1)
plt.fill_between(np.arange(256), 0, CDF)
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 255, step=50))
plt.title('Mine')
f.add_subplot(1,2,2)
plt.hist(npimageG.ravel(), 256, [0, 256], cumulative=True, density=True)
plt.xticks(np.arange(0, 255, step=50))
plt.title('Matplotlib')

# In order to check the first implementation, we can throw in an assertion
CDF_2 = plt.hist(npimageG.ravel(), 256, [0, 256], cumulative=True, density=True);

assert np.array_equal(CDF,CDF_2[0])

'''
Histogram Equalization
Is a method in image processing of contrast adjustment using the image's histogram. More information and examples on https://en.wikipedia.org/wiki/Histogram_equalization
'''
# First, let's declare our formula variables
CDF_min = np.min(CDF) # CDF Minimun value
MN = npimageG.shape[0] * npimageG.shape[1] # Total number of pixels
L = 256 # Number of grey levels used 

# Apply histogram equalization
hv = np.asarray([((x - CDF_min)/(MN - CDF_min))*(L - 1) for x in CDF])

#Create zeros-array with image dimensions
npimageHE = np.zeros((npimageG.shape[0], npimageG.shape[1]))

# Compute our new image
for row in range(npimageG.shape[0]):
    for col in range(npimageG.shape[1]):
        value = npimageG[row, col]
        npimageHE[row, col] = hv[int(value)]

# Normalize new image to [0 255]
npimageHE = npimageHE / np.max(npimageHE) * 255

# Let's Plot the images along with their respectively CDF
f = plt.figure(figsize=(15,15))
f.add_subplot(2,2,1)
plt.imshow(npimageG, cmap = 'gray')
plt.title('Original')
plt.axis('off')

f.add_subplot(2,2,2)
plt.hist(npimageG.ravel(), 256, [0, 256], cumulative=True, density=True)
plt.title('CDF')

f.add_subplot(2,2,3)
plt.imshow(npimageHE, cmap = 'gray')
plt.title('Histogram Equalization')   
plt.axis('off')

f.add_subplot(2,2,4)
plt.hist(npimageHE.ravel(), 256, [0, 256], cumulative=True, density=True)
plt.title('CDF')