#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('empire.jpg')
img_45 = cv.imread('empire_45.jpg')
img_zoomedout = cv.imread('empire_zoomedout.jpg')
img_another = cv.imread('fisherman.jpg')

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_45_gray = cv.cvtColor(img_45, cv.COLOR_BGR2GRAY)
img_zoomedout_gray = cv.cvtColor(img_zoomedout, cv.COLOR_BGR2GRAY)
img_another_gray = cv.cvtColor(img_another, cv.COLOR_BGR2GRAY)


# In[60]:



sift = cv.xfeatures2d.SIFT_create()

kp, des = sift.detectAndCompute(img_gray, None)
kp_45, des_45 = sift.detectAndCompute(img_45_gray, None)
kp_zoomedout, des_zoomedout = sift.detectAndCompute(img_zoomedout_gray, None)
kp_another, des_another = sift.detectAndCompute(img_another_gray, None)


# In[61]:


bf = cv.BFMatcher()
train = des_45
query = des
matches_des_des_45 = bf.match(query, train)


# In[62]:


matches_des_des_45 = sorted(matches_des_des_45, key = lambda x:x.distance)


# In[63]:



nBestMatches = 10
matching_des_des_45 = cv.drawMatches(img_gray, kp, img_45_gray, kp_45,
matches_des_des_45[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_des_45)


# In[112]:


kp_train = kp_45
kp_query = kp 
total =0
for i in range (0, nBestMatches):
    print("match ", i, " info")
    print("\tdistance:", matches_des_des_45[i].distance)
    print("\tkeypoint in train: ID:", matches_des_des_45[i].trainIdx, " x:",
          kp_train[matches_des_des_45[i].trainIdx].pt[0], " y:",
          kp_train[matches_des_des_45[i].trainIdx].pt[1])
    print("\tkeypoint in query: ID:", matches_des_des_45[i].queryIdx, " x:",
          kp_query[matches_des_des_45[i].queryIdx].pt[0], " y:",
          kp_query[matches_des_des_45[i].queryIdx].pt[1])
    total = total + matches_des_des_45[i].distance
    
print(total)


# In[65]:


matches_des_45_des = bf.match(des_45, des)
matches_des_45_des = sorted(matches_des_45_des, key = lambda x:x.distance)
matching_des_45_des = cv.drawMatches(img_45_gray, kp_45, img_gray, kp,
matches_des_45_des[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(matching_des_45_des)


# In[90]:


dm= cv.DescriptorMatcher_create('BruteForce')
train = des_45
query = des
matches_des_des_45 = dm.match(query, train)


# In[91]:


matches_des_des_45 = sorted(matches_des_des_45, key = lambda x:x.distance)


# In[92]:


nBestMatches = 10
matching_des_des_45 = cv.drawMatches(img_gray, kp, img_45_gray, kp_45,
matches_des_des_45[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_des_45)


# In[113]:


kp_train = kp_45
kp_query = kp
nBestMatches =19
for i in range (0, nBestMatches):
    print("match ", i, " info")
    print("\tdistance:", matches_des_des_45[i].distance)
    print("\tkeypoint in train: ID:", matches_des_des_45[i].trainIdx, " x:",
          kp_train[matches_des_des_45[i].trainIdx].pt[0], " y:",
          kp_train[matches_des_des_45[i].trainIdx].pt[1])
    print("\tkeypoint in query: ID:", matches_des_des_45[i].queryIdx, " x:",
          kp_query[matches_des_des_45[i].queryIdx].pt[0], " y:",
          kp_query[matches_des_des_45[i].queryIdx].pt[1])
    total = total + matches_des_des_45[i].distance
    
print(total)
    


# In[82]:


matches_des_45_des = bf.match(des_45, des)
matches_des_45_des = sorted(matches_des_45_des, key = lambda x:x.distance)
matching_des_45_des = cv.drawMatches(img_45_gray, kp_45, img_gray, kp,
matches_des_45_des[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(matching_des_45_des)


# In[93]:


dm= cv.DescriptorMatcher_create('BruteForce')
train = des_zoomedout
query = des
matches_des_des_z = dm.match(query, train)


# In[94]:


matches_des_des_z = sorted(matches_des_des_z, key = lambda x:x.distance)


# In[95]:



matching_des_des_z = cv.drawMatches(img_gray, kp, img_zoomedout_gray, kp_zoomedout,
matches_des_des_z[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_des_z)


# In[114]:


kp_train = kp_zoomedout
kp_query = kp
nBestMatches = 21
for i in range (0, nBestMatches):
    print("match ", i, " info")
    print("\tdistance:", matches_des_des_z[i].distance)
    print("\tkeypoint in train: ID:", matches_des_des_z[i].trainIdx, " x:",
          kp_train[matches_des_des_z[i].trainIdx].pt[0]," y:",
          kp_train[matches_des_des_z[i].trainIdx].pt[1])
    print("\tkeypoint in query: ID:", matches_des_des_z[i].queryIdx, " x:",
          kp_query[matches_des_des_z[i].queryIdx].pt[0], " y:",
          kp_query[matches_des_des_z[i].queryIdx].pt[1])
    total = total + matches_des_des_45[i].distance
    
print(total)


# In[116]:


matches_des_z_des = bf.match(des_zoomedout, des)
matches_des_z_des = sorted(matches_des_z_des, key = lambda x:x.distance)
matching_des_z_des = cv.drawMatches(img_zoomedout_gray, kp_zoomedout, img_gray, kp,
matches_des_z_des[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(matching_des_z_des)


# In[75]:


dm= cv.DescriptorMatcher_create('BruteForce')
train = des_another
query = des
matches_des_des_a = dm.match(query, train)


# In[76]:


matches_des_des_a = sorted(matches_des_des_a, key = lambda x:x.distance)


# In[77]:


nBestMatches = 10
matching_des_des_a = cv.drawMatches(img_gray, kp, img_another_gray, kp_another,
matches_des_des_a[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matching_des_des_a)


# In[115]:


kp_train = kp_another
kp_query = kp
nBestMatches =20
for i in range (0, nBestMatches):
    print("match ", i, " info")
    print("\tdistance:", matches_des_des_a[i].distance)
    print("\tkeypoint in train: ID:", matches_des_des_a[i].trainIdx, " x:",
          kp_train[matches_des_des_a[i].trainIdx].pt[0], " y:",
          kp_train[matches_des_des_a[i].trainIdx].pt[1])
    print("\tkeypoint in query: ID:", matches_des_des_a[i].queryIdx, " x:",
          kp_query[matches_des_des_a[i].queryIdx].pt[0], " y:",
          kp_query[matches_des_des_a[i].queryIdx].pt[1])
    total = total + matches_des_des_45[i].distance
    
print(total)


# In[79]:


matches_des_a_des = bf.match(des_another, des)
matches_des_a_des = sorted(matches_des_a_des, key = lambda x:x.distance)
matching_des_a_des = cv.drawMatches(img_another_gray, kp_another, img_gray, kp,
matches_des_a_des[:nBestMatches],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(matching_des_a_des)


# In[ ]:




