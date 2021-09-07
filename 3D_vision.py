#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#setup camera with a simple camera matrix P
f = 100
cx = 200
cy = 200
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
I = np.eye(3) #i.e., R
t = np.array([[0], [0], [0]])
P = np.dot(K, np.hstack((I, t)))


# In[2]:


def project(P, X): #X is an array of 3D points
    x = np.dot(P, X)
    for i in range(3): #convert to inhomogeneous coordinates
        x[i] /= x[2]
    return x


# In[3]:


points_3D = np.loadtxt('house.p3d').T #T means tranpose
points_3D = np.vstack((points_3D, np.ones(points_3D.shape[1])))


# In[4]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(figsize = [15,15])
ax = fig.gca(projection = "3d")
ax.view_init(elev = None, azim = None) #you can set elevation and azimuth with different values
ax.plot(points_3D[0], points_3D[1], points_3D[2], 'o')
plt.draw()
plt.show()


# In[5]:


#projection
points_2D = project(P, points_3D)
#plot projection
from matplotlib import pyplot as plt
plt.plot(points_2D[0], points_2D[1], 'k.')
plt.show()


# In[6]:


print(points_2D.shape)
print(points_3D.shape)


# In[7]:


n_points = 6
points_3D_sampled = points_3D[:,:n_points]
points_2D_sampled = points_2D[:,:n_points]
points_3D_sampled.shape


# In[8]:


A = np.zeros((2*n_points, 12), np.float32)
print(A)


# In[9]:


for i in range(n_points):
    A[2*i,:4] = points_3D_sampled[:,i].T
    A[2*i,8:12] = -points_2D_sampled[0,i] * points_3D_sampled[:,i].T
    A[2*i+1,4:8] = points_3D_sampled[:,i].T
    A[2*i+1,8:12] = -points_2D_sampled[1,i] * points_3D_sampled[:,i].T


# In[10]:


A.shape
print(A)


# In[11]:


from scipy import linalg
p = linalg.solve(A, np.zeros((12, 1), np.float32))
print(p)


# In[12]:


U, S, V = linalg.svd(A)


# In[13]:


minS = np.min(S)
conditon = (S == minS)
minID = np.where(conditon)
print('index of the smallest singular value is: ', minID[0])


# In[14]:


P_hat = V[minID[0],:].reshape(3, 4) / minS


# In[15]:


print(P)
print(P_hat)


# In[16]:


x_P_hat = project(P_hat, points_3D_sampled[:, 0])
print(x_P_hat)


# In[17]:


x_P = points_2D_sampled[:,0]
print(x_P)


# In[18]:


x_P = points_2D
x_P_hat = project(P_hat, points_3D)

dist = 0
for i in range(x_P.shape[1]):
    dist += np.linalg.norm(x_P[:,i] - x_P_hat[:,i])
dist /= x_P.shape[1]
print(dist)


# In[ ]:





# In[19]:


import homography
import sfm
import ransac


# In[20]:


import cv2 as cv
sift = cv.xfeatures2d.SIFT_create()
img1 = cv.imread('alcatraz1.jpg')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(img1_gray, None)
img2 = cv.imread('alcatraz2.jpg')
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
img1_kp = img1.copy()
img1_kp = cv.drawKeypoints(img1, kp1, img1_kp)
print("Number of detected keypoints in img1: %d" % (len(kp1)))
img2_kp = img2.copy()
img2_kp = cv.drawKeypoints(img2, kp2, img2_kp)
print("Number of detected keypoints in img2: %d" % (len(kp2)))
img1_2_kp = np.hstack((img1_kp, img2_kp))
plt.figure(figsize = (20, 10))
plt.imshow(img1_2_kp[:,:,::-1])
plt.axis('off')


# In[21]:


bf = cv.BFMatcher(crossCheck = True) #crossCheck = True means we want to find consistent matches
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
print("Number of consistent matches: %d" % len(matches))


# In[22]:


img1_2_matches = cv.drawMatches(img1, kp1, img2, kp2,
matches[:20],
None,
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize = (20, 10))
plt.imshow(img1_2_matches[:,:,::-1])
plt.axis('off')


# In[23]:


n_matches = 1000
kp1_array = np.zeros((2, n_matches), np.float32)
for i in range(n_matches):
    kp1_array[0][i] = kp1[matches[i].queryIdx].pt[0]
    kp1_array[1][i] = kp1[matches[i].queryIdx].pt[1]
kp2_array = np.zeros((2, n_matches), np.float32)
for i in range(n_matches):
    kp2_array[0][i] = kp2[matches[i].trainIdx].pt[0]
    kp2_array[1][i] = kp2[matches[i].trainIdx].pt[1]


# In[24]:


x1 = homography.make_homog(kp1_array)
x2 = homography.make_homog(kp2_array)


# In[25]:


K = np.array([[2394,0,932], [0,2398,628], [0,0,1]])
P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])


# In[26]:


x1n = np.dot(linalg.inv(K), x1)
x2n = np.dot(linalg.inv(K), x2)


# In[27]:


#estimate E with RANSAC
model = sfm.RansacModel()
E, inliers = sfm.F_from_ransac(x1n, x2n, model)


# In[28]:


#compute camera matrices (P2 will be list of four solutions)
P2_all = sfm.compute_P_from_essential(E)
#pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
#triangulate inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2_all[i])
    d1 = np.dot(P1, X)[2]
    d2 = np.dot(P2_all[i], X)[2]
    s = sum(d1 > 0) + sum(d2 > 0)
    if s > maxres:
        maxres = s
        ind = i
        infront = (d1 > 0) & (d2 > 0)
P2 = P2_all[ind]


# In[29]:


X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2)
X = X[:, infront]


# In[30]:


print(len(X[0]))


# In[31]:


fig = plt.figure(figsize = [20,20])
ax = fig.gca(projection = "3d")
ax.view_init(elev = None, azim = None) #you can set elevation and azimuth with different values
ax.plot(X[0], X[1], X[2], 'o')
plt.draw()
plt.show()


# In[32]:


prj= project(P2, X)
prj = np.dot(K, prj)


# In[33]:


plt.figure()
plt.imshow(img2[:,:,::-1])
plt.plot(x2[0], x2[1], 'b.')


# In[34]:


plt.figure()
plt.imshow(img2[:,:,::-1])
plt.plot(prj[0], prj[1], 'r.')


# In[ ]:





# In[ ]:





# In[ ]:




