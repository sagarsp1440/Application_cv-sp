#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv


# In[2]:


img1=cv.imread('img1.jpg')
img2=cv.imread('img2.jpg')
img3=cv.imread('img3.jpg')


# In[3]:


img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img3_gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)


# In[4]:


hist1_gray = cv.calcHist([img1_gray],[0],None,[256],[0,256])
hist2_gray = cv.calcHist([img2_gray],[0],None,[256],[0,256])
hist3_gray = cv.calcHist([img3_gray],[0],None,[256],[0,256])


# In[5]:


def CH(P,Q):
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon
    distance = np.sum(np.square(P-Q)/(P+Q))
    return distance
print(CH(hist1_gray,hist2_gray))
print(CH(hist2_gray,hist3_gray))
print(CH(hist1_gray,hist3_gray))


# In[6]:


hist1_gray = cv.calcHist([img1_gray],[0],None,[256],[0,255])
hist2_gray = cv.calcHist([img2_gray],[0],None,[256],[0,255])
hist3_gray = cv.calcHist([img3_gray],[0],None,[256],[0,255])


# In[7]:


from matplotlib import pyplot as plt
plt.plot(hist1_gray, label = 'img1_gray')
plt.plot(hist2_gray, label = 'img2_gray')
plt.plot(hist3_gray, label = 'img3_gray')
plt.legend(loc="upper right")
plt.xlim([0,256])

plt.show()


# In[8]:


i=0
hist1_gray_sum= hist1_gray.sum()
hist2_gray_sum= hist2_gray.sum()
hist3_gray_sum= hist3_gray.sum()
for i in range(256):
    hist1_gray[i]=hist1_gray[i]/hist1_gray_sum
    hist2_gray[i]=hist2_gray[i]/hist2_gray_sum
    hist3_gray[i]=hist3_gray[i]/hist3_gray_sum
    


# In[9]:


print(hist1_gray.sum())


# In[10]:


from matplotlib import pyplot as plt
plt.plot(hist1_gray, label = 'img1_gray')
plt.plot(hist2_gray, label = 'img2_gray')
plt.plot(hist3_gray, label = 'img3_gray')
plt.legend(loc="upper right")
plt.xlim([0,256])

plt.show()


# In[11]:


#Addsing epsilon so as to eliminate any possible null values
def KL(P,Q):
    epsilon = 0.00001
    P = P+epsilon
    Q = Q+epsilon
    divergence = np.sum(P*np.log(P/Q))
    return divergence


# In[12]:


print('For img 1 and img 2')
print( KL(hist1_gray, hist2_gray)+KL(hist2_gray, hist1_gray))
print('For img 2 and img 3')
print( KL(hist2_gray, hist3_gray)+KL(hist3_gray, hist2_gray))
print('For img 1 and img 3')
print( KL(hist1_gray, hist3_gray)+KL(hist3_gray, hist1_gray))


# For chi squared method the lower the value, more similar the image are comparitively.

# For KL divergence the higher the value, more similar the images are comparitively i.e. img2 and img3 are more similar
