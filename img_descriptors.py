#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('empire.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(img_gray, 'gray')


# In[42]:


local_region_size = 3
kernel_size = 3 
k = 0.04 
threshold = 500.0


# In[43]:


img_gray = np.float32(img_gray)


# In[44]:


img_gray


# In[84]:


Harris_res_img = cv.cornerHarris(img_gray, local_region_size, kernel_size, k)
plt.imshow(Harris_res_img, 'gray')
cv.imwrite('Harris.png',Harris_res_img)


# In[46]:


highlighted_colour = [0, 0, 255]
highlighted_img = img.copy()
highlighted_img[Harris_res_img > threshold] = highlighted_colour
plt.imshow(highlighted_img[:,:,::-1]) 


# In[47]:


height, width = Harris_res_img.shape
count=0


# In[48]:


for i in range(0, height):
    for j in range(0, width):
        if(Harris_res_img[i,j]>threshold):
            count=count+1 


# In[49]:


print("the number of detected corners ",count)


# In[52]:


threshold = 0.001 * Harris_res_img.max()


# In[53]:


print(threshold)


# In[54]:


sift = cv.xfeatures2d.SIFT_create()


# In[55]:


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
kp = sift.detect(img_gray, None)


# In[86]:


img_gray_kp = img_gray.copy()
img_gray_kp = cv.drawKeypoints(img_gray, kp, img_gray_kp)
plt.imshow(img_gray_kp)
cv.imwrite('SIFT.png',img_gray_kp)
print("Number of detected keypoints: %d" % (len(kp)))


# In[ ]:





# In[59]:


img_gray_kp = cv.drawKeypoints(img_gray, kp, img_gray_kp, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_gray_kp)


# In[60]:


kp, des = sift.compute(img_gray, kp)


# In[61]:


print(des.shape)


# In[62]:


img_45 = cv.imread('empire_45.jpg')
img_zoomedout = cv.imread('empire_zoomedout.jpg')
img_45_gray = cv.cvtColor(img_45, cv.COLOR_BGR2GRAY)
img_zoomedout_gray = cv.cvtColor(img_zoomedout, cv.COLOR_BGR2GRAY)
img_another = cv.imread('fisherman.jpg')
img_another_gray = cv.cvtColor(img_another, cv.COLOR_BGR2GRAY)


# In[63]:


kp_45, des_45 = sift.detectAndCompute(img_45_gray, None)
kp_zoomedout, des_zoomedout = sift.detectAndCompute(img_zoomedout_gray, None)
kp_another, des_another = sift.detectAndCompute(img_another_gray, None)


# In[64]:


print("The number of keypoints in img_gray is %d" % (len(des)))
print("The number of keypoints in img_45_gray is %d" % (len(des_45)))


# In[65]:


print("The number of keypoints in img_gray is %d" % (len(des)))
print("The number of keypoints in img_45_gray is %d" % (len(des_zoomedout)))


# In[66]:


print("The number of keypoints in img_gray is %d" % (len(des)))
print("The number of keypoints in img_45_gray is %d" % (len(des_another)))


# In[67]:


from scipy.spatial.distance import directed_hausdorff


# In[68]:


u=np.array(des)


# In[69]:


v=np.array(des_45)


# In[73]:


y=np.array(des_zoomedout)


# In[74]:


z=np.array(des_another)


# In[79]:


print("hausdoff distance between des and des_45",max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0]))


# In[82]:


print("hausdoff distance between des and des_Zoomedout",max(directed_hausdorff(u, y)[0], directed_hausdorff(y, u)[0]))


# In[83]:


print("hausdoff distance between des and des_another",max(directed_hausdorff(u, z)[0], directed_hausdorff(z, u)[0]))

