#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
img = cv.imread('img1.jpg')


# In[2]:


cv.imshow('input',img)
cv.waitKey(0)
cv.destroyAllWindows()


# In[3]:


hav_img=cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('hsvoutput',hav_img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('hav_img.png',hav_img)


# In[4]:


gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grayoutput',gray_img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('gray_img.png',gray_img)


# In[5]:


height, width = img.shape[:2]
print(height,width)


# In[6]:


h_scale, V_scale= 0.6, 0.5


# In[7]:


new_height= (int)(height*V_scale)
new_width= (int)(width*h_scale)
rsz_img= cv.resize(img, (new_height,new_width), interpolation= cv.INTER_LINEAR)
cv.imshow('resize image',rsz_img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('resizre_img.png',rsz_img)


# In[8]:


t_x=160
t_y=220


# In[9]:


M = np.float32([[1, 0, t_x], [0, 1, t_y]])


# In[10]:


height, width = img.shape[:2] 
img_translation = cv.warpAffine(img, M, (width, height))
cv.imshow('translation img', img_translation)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('translate img.png',img_translation)


# In[11]:


theta= 60


# In[12]:


c_x = (width - 1) / 2.0
c_y = (height - 1) / 2.0 
c = (c_x, c_y)
print(c)


# In[14]:


s = 1
M = cv.getRotationMatrix2D(c, theta, s)


# In[16]:


img_rotation = cv.warpAffine(img, M, (width, height))
cv.imshow('rotation', img_rotation)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('rotated img.png',img_rotation)


# In[17]:


m00 = 0.35
m01 = 0.30
m02 = -45.1
m10 = -0.13
m11 = 0.6
m12 = 300
M = np.float32([[m00, m01, m02], [m10, m11, m12]])


# In[18]:


height, width = img.shape[:2]
img_affine = cv.warpAffine(img, M, (width, height))
cv.imshow('affine', img_affine)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('Affine img.png',img_affine)


# In[ ]:


# Both warpaffine and resize can be used to scale an image, but can be dependant on the scaling algorithm used

