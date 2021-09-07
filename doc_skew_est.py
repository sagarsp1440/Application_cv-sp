#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt
img = cv.imread('fisherman.jpg') 
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# In[2]:


plt.imshow(img_gray)


# In[3]:


height, width = img_gray.shape
ori_img_gray = np.zeros((height, width), np.float32)
ori_img_gray_ij=np.array((height,width),np.float32)


# In[4]:


ori_img_gray.shape


# In[5]:


D_x = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
der_x = cv.filter2D(img_gray, -1, D_x)


# In[6]:


der_x


# In[7]:


D_y = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
der_y = cv.filter2D(img_gray, -1, D_y)


# In[8]:


der_y


# In[9]:


for i in range(0, height):
    for j in range(0, width):
        if((der_x[i,j] & der_y[i,j])==0):
            ori_img_gray[i,j]=math.inf
        else:
            ori_img_gray[i,j]=((math.atan2(der_x[i,j],der_y[i,j]))*180)/math.pi
        if(ori_img_gray[i,j]!=math.inf):
            ori_img_gray[i,j]=ori_img_gray[i,j]+90



# In[ ]:





# In[10]:


hist_1= cv.calcHist([ori_img_gray],[0],None,[181],[0,180])


# In[11]:


from matplotlib import pyplot as plt
plt.plot(hist_1,color='r')
plt.xlim([0,180])
plt.show()


# In[45]:


histogram, bin_edges = np.histogram([ori_img_gray], bins=256, range=(0, 180))
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, 180.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()


# In[13]:


img1 = cv.imread('empire.jpg') 
img_gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)


# In[14]:


plt.imshow(img_gray1)


# In[15]:


height, width = img_gray1.shape
ori_img_gray1 = np.zeros((height, width), np.float32)


# In[16]:


img_gray1.shape


# In[17]:


D_x = np.float32([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
der_x = cv.filter2D(img_gray1, -1, D_x)


# In[18]:


D_y = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
der_y = cv.filter2D(img_gray1, -1, D_y)


# In[19]:


for i in range(0, height):
    for j in range(0, width):
        if((der_x[i,j] & der_y[i,j])==0):
            ori_img_gray1[i,j]=math.inf
        else:
            ori_img_gray1[i,j]=((math.atan2(der_x[i,j],der_y[i,j]))*180)/math.pi
        if(ori_img_gray1[i,j]!=math.inf):
            ori_img_gray1[i,j]=ori_img_gray1[i,j]+90


# In[20]:


hist_2= cv.calcHist([ori_img_gray1],[0],None,[181],[0,180])


# In[21]:


plt.plot(hist_2,color='b')
plt.xlim([0,180])
plt.show()


# In[29]:


#doc = cv.imread('doc.jpg', 0) 


# In[30]:


#for Task 2 applying it on doc_1
doc=cv.imread('doc_1.jpg',0)


# In[31]:


threshold = 200
ret, doc_bin = cv.threshold(doc, threshold, 255, cv.THRESH_BINARY)


# In[32]:


closing_se = np.ones((15, 1), np.int)


# In[33]:


doc_bin = 255 - doc_bin


# In[34]:


closing = cv.morphologyEx(doc_bin, cv.MORPH_CLOSE, closing_se)
plt.imshow(closing, 'gray')
cv.imwrite('closing.png',closing)


# In[35]:


opening_se = np.ones((8, 8), np.int)


# In[36]:


opening = cv.morphologyEx(closing, cv.MORPH_OPEN, opening_se)
plt.imshow(opening, 'gray')
cv.imwrite('opening.png',opening)


# In[37]:


num_labels, labels_im = cv.connectedComponents(opening)


# In[38]:


def ExtractConnectedComponents(num_labels, labels_im):
    connected_components = [[] for i in range(0, num_labels)]
    height, width = labels_im.shape
    for i in range(0, height):
        for j in range(0, width):
            if labels_im[i, j] >= 0:
                connected_components[labels_im[i, j]].append((j, i))
    return connected_components


# In[39]:


connected_components = ExtractConnectedComponents(num_labels, labels_im)


# In[40]:


import math
def FindOrientation(cc):
    mx = 0
    my = 0
    mxx = 0
    myy = 0
    mxy = 0
    for i in range(0, len(cc)):
        mx += cc[i][0] # cc[i][0] is used to store the x coordinate of pixel cc[i]
        my += cc[i][1] # cc[i][1] is used to store the y coordinate of pixel cc[i]
    mx /= len(cc)
    my /= len(cc)
    for i in range(0, len(cc)):
        dx = cc[i][0] - mx
        dy = cc[i][1] - my
        mxx += (dx * dx)
        myy += (dy * dy)
        mxy += (dx * dy)
    mxx /= len(cc)
    myy /= len(cc)
    mxy /= len(cc)
        
    theta = - math.atan2(2 * mxy, mxx - myy) / 2
    return theta


# In[41]:


orientations = np.zeros(num_labels, np.float32)
for i in range(0, num_labels):
    orientations[i] = FindOrientation(connected_components[i])


# In[42]:


import statistics
orientation = statistics.median(orientations)


# In[43]:


height, width = doc.shape
c_x = (width - 1) / 2.0 # column index varies in [0, width-1]
c_y = (height - 1) / 2.0 # row index varies in [0, height-1]
c = (c_x, c_y) # A point is defined by x and y coordinate
M = cv.getRotationMatrix2D(c, -orientation * 180 / math.pi, 1)
doc_deskewed = cv.warpAffine(doc, M, (width, height))
plt.imshow(doc_deskewed, 'gray')
cv.imwrite('deskewd_doc.png',doc_deskewed)

