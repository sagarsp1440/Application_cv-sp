#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv


# In[2]:


img= cv.imread('img1.jpg')
cv.imshow('img1.jpg',img)
cv.waitKey(0)
cv.destroyAllWindows()


# In[3]:


hist_blue= cv.calcHist([img],[0],None,[256],[0,256])


# In[4]:


from matplotlib import pyplot as plt
plt.plot(hist_blue,color='b')
plt.xlim([0,256])
plt.show()


# In[5]:


hist_green= cv.calcHist([img],[1],None,[256],[0,256])


# In[6]:


from matplotlib import pyplot as plt
plt.plot(hist_green,color='g')
plt.xlim([0,256])
plt.show()


# In[7]:


hist_red= cv.calcHist([img],[2],None,[256],[0,256])


# In[8]:


from matplotlib import pyplot as plt
plt.plot(hist_red,color='r')
plt.xlim([0,256])
plt.show()


# In[9]:


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# In[10]:


hist_gray = cv.calcHist([img_gray],[0],None,[256],[0,256])
plt.plot(hist_gray,color='gray')
plt.xlim([0,256])
plt.show()


# In[11]:


def getCummulativeDis(hist):
    c = [] #cummulative distribution
    s = 0
    for i in range(0, len(hist)):
        s = s + hist[i]
        c.append(s)
    return c


# In[12]:


c = getCummulativeDis(hist_gray)
plt.plot(c, label = 'cummulative distribution', color = 'orange')
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()


# In[13]:


img_eq=cv.equalizeHist(img_gray)


# In[14]:


hist_eq=cv.calcHist([img_eq],[0],None,[256],[0,256])
plt.plot(hist_eq)
plt.xlim([0,256])
plt.show()


# In[15]:


c_eq = getCummulativeDis(hist_eq)
plt.plot(c_eq, label = 'cummulative distribution after hist equalisation',color='orange')
plt.legend(loc="upper left")
plt.xlim([0,256])
plt.show()


# In[16]:


img_histeq = np.hstack((img_gray, img_eq)) 
cv.imwrite('img_histeq.png', img_histeq)


# Adaptive Histogram Equalization
# 

# In[ ]:


img = cv.imread('img1.jpg',0)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imshow('clah',cl1)
cv.waitKey(0)
cv.destroyAllWindows()

img_hist_stack=np.hstack((img_eq,cl1))
cv.imwrite('img_hist_types.png',img_hist_stack)


# In[ ]:




