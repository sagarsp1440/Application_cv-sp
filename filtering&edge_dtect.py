#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('empire.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
avg_kernel = np.ones((5,5), np.float32) / 25 
avg_result = cv.filter2D(img_gray, -1, avg_kernel)
plt.imshow(img_gray, 'gray')
cv.imwrite('empire_gray.png',img_gray)


# In[3]:


sobel_filter = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
sobel_filter = cv.filter2D(img_gray, -1, sobel_filter)
plt.imshow(sobel_filter, 'gray')
cv.imwrite('sobel_filter.png',sobel_filter)


# In[5]:


corner_filter = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4
corner_filter = cv.filter2D(img_gray, -1, corner_filter)
plt.imshow(corner_filter, 'gray')
cv.imwrite('corner_filter.png',corner_filter)


# In[6]:


gauss_filter = np.float32([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
gauss_filter = cv.filter2D(img_gray, -1, gauss_filter)
plt.imshow(gauss_filter, 'gray')
cv.imwrite('gauss_filter.png',gauss_filter)


# In[14]:


img_noise = cv.imread('empire_shotnoise.jpg')
img_noise_gray = cv.cvtColor(img_noise, cv.COLOR_BGR2GRAY)
ksize = 7
med_result = cv.medianBlur(img_noise_gray, ksize)
plt.imshow(med_result, 'gray')
cv.imwrite('Median_filter.png',med_result)


# In[15]:


rad = 9 
sigma_s = 25 
sigma_c = 40 
bil_result = cv.bilateralFilter(img_noise_gray, rad, sigma_c, sigma_s)
plt.imshow(bil_result, 'gray')
cv.imwrite('Bilateral_filter.png',bil_result)


# In[24]:


D_x = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
der_x = cv.filter2D(img_gray, -1, D_x)
plt.imshow(der_x, 'gray')
cv.imwrite('horizontal_der.png',der_x)


# In[25]:


D_y = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
der_y = cv.filter2D(img_gray, -1, D_y)
plt.imshow(der_y, 'gray')
cv.imwrite('vertical_der.png',der_y)


# In[26]:


import math
height, width = img_gray.shape
mag_img_gray = np.zeros((height, width), np.float32) #gradient magnitude of img_gray
for i in range(0, height):
    for j in range(0, width):
        square_der_x = float(der_x[i, j]) * float(der_x[i, j])
        square_der_y = float(der_y[i, j]) * float(der_y[i, j])
        mag_img_gray[i, j] = int(math.sqrt(square_der_x + square_der_y))
plt.imshow(mag_img_gray,'gray')
cv.imwrite('gradient_mag.png',mag_img_gray)


# In[27]:


minVal = 50 
maxVal = 100 
Canny_edges = cv.Canny(img_gray, minVal, maxVal)
plt.imshow(Canny_edges, 'gray')
cv.imwrite('Canny_edg.png',Canny_edges)


# In[ ]:




