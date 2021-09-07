#!/usr/bin/env python
# coding: utf-8

# # Task 4.3p -   Image Recognition - LEGO Bricks
# **Authors:** Justin Tomlinson & Sagar Prakesh<br>
# **Class:** SIT879 - Applications of Computer Vision and Speech Processing

# ## Prepare dataset

# In[ ]:


import os
import cv2 as cv
import random
import numpy as np
import pandas as pd

import splitfolders


# In[ ]:


bricks = ['brick_2x2', 'brick_corner_1x2x2', 'flat_tile_1x2', 'plate_1x1']
path = './4.3_images/'

file_names = []
brick_labels = []


for i in range(0, len(bricks)):
    sub_path = path + bricks[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_brick_labels = [i] * len(sub_file_names)
    file_names += sub_file_names
    brick_labels += sub_brick_labels


# All the images in the lego dataset are uniform dimensions and lighting so i change the size and brightness of a random subset of images to get some variation.

# In[ ]:


nb_images_to_alter = int(len(file_names)*0.5)

images_to_alter = random.sample(file_names, nb_images_to_alter)
images_to_resize = random.sample(images_to_alter, int(len(file_names)*0.4))
images_to_brighten = random.sample(images_to_alter, int(len(file_names)*0.2))

img_scales = [0.5, 0.6, 0.7, 0.8]


# In[ ]:


for file in images_to_resize:
    img = cv.imread(file, cv.IMREAD_UNCHANGED)
    scale_factor = random.sample(img_scales, 1)[0]
    img = cv.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
    cv.imwrite(file, img)


# In[ ]:


for file in images_to_brighten:
    img = cv.imread(file, cv.IMREAD_UNCHANGED)
    scale_factor = random.sample(img_scales, 1)[0]
    img = cv.convertScaleAbs(img, alpha=1.3, beta=35)
    cv.imwrite(file, img)


# In[ ]:


splitfolders.ratio("./4.3_images/", output="./4.3_images/dataset/", seed=42, ratio=(.4, .3, .3), group_prefix=None)


# ## Model Build

# In[1]:


import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pycm import *


# In[2]:


def plot_conf_matrix(data, labels, Title=None):
    plt.figure(figsize=(4,3.5))
    plt.title(Title)
    chart = sns.heatmap(data, cmap='OrRd', square=True,
                        annot=True, linewidths=1, linecolor='white',
                        yticklabels=labels, xticklabels=labels, fmt='.3f',
                        cbar_kws={'fraction' : 0.1})
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    chart.set_yticklabels(chart.get_yticklabels(), rotation=45, horizontalalignment='right')
    plt.show


# In[3]:


class Dictionary(object):
    def __init__(self, name, img_filenames, num_words):
        self.name = name #name of your dictionary
        self.img_filenames = img_filenames #list of image filenames
        self.num_words = num_words #the number of words

        self.training_data = [] #this is the training data required by the K-Means algorithm
        self.words = [] #list of words, which are the centroids of clusters

    def learn(self):
        sift = cv.xfeatures2d.SIFT_create()

        num_keypoints = [] #this is used to store the number of keypoints in each image

        #load training images and compute SIFT descriptors
        for filename in self.img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            list_des = sift.detectAndCompute(img_gray, None)[1]
            if list_des is None:
                num_keypoints.append(0)
            else:
                num_keypoints.append(len(list_des))
                for des in list_des:
                    self.training_data.append(des)

        #cluster SIFT descriptors using K-means algorithm
        kmeans = KMeans(self.num_words)
        kmeans.fit(self.training_data)
        self.words = kmeans.cluster_centers_

        #create word histograms for training images
        training_word_histograms = [] #list of word histograms of all training images
        index = 0
        for i in range(0, len(self.img_filenames)):
            #for each file, create a histogram
            histogram = np.zeros(self.num_words, np.float32)
            #if some keypoints exist
            if num_keypoints[i] > 0:
                for j in range(0, num_keypoints[i]):
                    histogram[kmeans.labels_[j + index]] += 1
                index += num_keypoints[i]
                histogram /= num_keypoints[i]
                training_word_histograms.append(histogram)

        return training_word_histograms

    def create_word_histograms(self, img_filenames):
        sift = cv.xfeatures2d.SIFT_create()
        histograms = []

        for filename in img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            descriptors = sift.detectAndCompute(img_gray, None)[1]

            histogram = np.zeros(self.num_words, np.float32) #word histogram for the input image

            if descriptors is not None:
                for des in descriptors:
                    #find the best matching word
                    min_distance = 1111111 #this can be any large number
                    matching_word_ID = -1 #initial matching_word_ID=-1 means no matching

                    for i in range(0, self.num_words): #search for the best matching word
                        distance = np.linalg.norm(des - self.words[i])
                        if distance < min_distance:
                            min_distance = distance
                            matching_word_ID = i

                    histogram[matching_word_ID] += 1

                histogram /= len(descriptors) #normalise histogram to frequencies

            histograms.append(histogram)

        return histograms


# In[4]:


import os

bricks = ['brick_2x2', 'brick_corner_1x2x2', 'flat_tile_1x2', 'plate_1x1']
path = './4.3_images/dataset/'

training_file_names = []
training_labels = []

test_file_names = []
test_labels = []

val_file_names = []
val_labels = []


for i in range(0, len(bricks)):
    sub_path = path + 'train/' + bricks[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_labels = [i] * len(sub_file_names)
    training_file_names += sub_file_names
    training_labels += sub_labels
    
for i in range(0, len(bricks)):
    sub_path = path + 'test/' + bricks[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_labels = [i] * len(sub_file_names)
    test_file_names += sub_file_names
    test_labels += sub_labels
    
for i in range(0, len(bricks)):
    sub_path = path + 'val/' + bricks[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_labels = [i] * len(sub_file_names)
    val_file_names += sub_file_names
    val_labels += sub_labels


# ## Selecting the right k-value for the BoW model

# In[5]:


num_keypoints = []
training_data = []

sift = cv.xfeatures2d.SIFT_create()

for filename in training_file_names:
        img = cv.imread(filename)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        list_des = sift.detectAndCompute(img_gray, None)[1]
        if list_des is None:
            num_keypoints.append(0)
        else:
            num_keypoints.append(len(list_des))
            for des in list_des:
                training_data.append(des)


# In[ ]:


sse = []
list_k = list(range(10, 200,10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(training_data)
    sse.append(km.inertia_)


# In[68]:


# Plot sse against k
plt.figure(figsize=(12, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


# ## Build the BoW model and get the word historgrams

# In[49]:


num_words = 125
dictionary_name = 'bricks'
dictionary = Dictionary(dictionary_name, training_file_names, num_words)


# In[50]:


training_word_histograms = dictionary.learn()


# In[9]:


import pickle

with open('brick_dictionary.dic', 'wb') as f:
    pickle.dump(dictionary, f)


# In[10]:


with open('brick_dictionary.dic', 'rb') as f: #'rb' is for binary read
    dictionary = pickle.load(f)


# In[51]:


test_word_histograms = dictionary.create_word_histograms(test_file_names)
validation_word_histograms = dictionary.create_word_histograms(val_file_names)


# ## 2. k-NN Optimisation

# In[52]:


from sklearn.neighbors import KNeighborsClassifier


# In[53]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


# In[54]:


ks =  [1, 3, 5, 7, 10, 13]

for k in ks:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(training_word_histograms, training_labels)
    predicted_labels = knn.predict(validation_word_histograms)
    cm = confusion_matrix(val_labels, predicted_labels, normalize='true')
    plot_conf_matrix(cm, bricks, f'k={k}, Val Acc.={accuracy_score(val_labels, predicted_labels)*100:.2f}%')


# ## K-NN Test set

# In[55]:


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(training_word_histograms, training_labels)
predicted_labels = knn.predict(test_word_histograms)
cm = confusion_matrix(test_labels, predicted_labels, normalize='true')
plot_conf_matrix(cm, bricks, f'k={1}, Test Acc.={accuracy_score(test_labels, predicted_labels)*100:.2f}%')


# ## 3. SVM optimisation

# In[56]:


from sklearn import svm


# In[57]:


cs = [10, 20, 30, 40, 50, 75]


# In[58]:


for c in cs:
    svm_classifier = svm.SVC(C = c,
                         kernel = 'linear', random_state=42)
    svm_classifier.fit(training_word_histograms, training_labels)
    predicted_labels = svm_classifier.predict(validation_word_histograms)
    cm = confusion_matrix(val_labels, predicted_labels, normalize='true')
    plot_conf_matrix(cm, bricks, f'C={c}, Val Acc.={accuracy_score(val_labels, predicted_labels)*100:.2f}%')


# ## SVM Test set

# In[59]:


svm_classifier = svm.SVC(C = 30,
                     kernel = 'linear', random_state=42)
svm_classifier.fit(training_word_histograms, training_labels)
predicted_labels = svm_classifier.predict(test_word_histograms)
cm = confusion_matrix(test_labels, predicted_labels, normalize='true')
plot_conf_matrix(cm, bricks, f'C={40}, Test Acc.={accuracy_score(test_labels, predicted_labels)*100:.2f}%')


# ## 4.. AdaBoost Optimisation

# In[60]:


from sklearn.ensemble import AdaBoostClassifier


# In[61]:


n_estmators =  [150, 200, 250, 300, 400, 500]


# Validation testing parameters

# In[62]:


for n in n_estmators:
    adb_classifier = AdaBoostClassifier(n_estimators = n,
                                        random_state = 42)

    adb_classifier.fit(training_word_histograms, training_labels)
    predicted_labels = adb_classifier.predict(validation_word_histograms)
    cm = confusion_matrix(val_labels, predicted_labels, normalize='true')
    plot_conf_matrix(cm, bricks, f'n_estmators={n}, Val Acc.={accuracy_score(val_labels, predicted_labels)*100:.2f}%')


# ## AdaBoost Test set

# In[63]:


adb_classifier = AdaBoostClassifier(n_estimators = 500,
                                    random_state = 42)

adb_classifier.fit(training_word_histograms, training_labels)
predicted_labels = adb_classifier.predict(test_word_histograms)
cm = confusion_matrix(test_labels, predicted_labels, normalize='true')
plot_conf_matrix(cm, bricks, f'n_estmators={500}, test Acc.={accuracy_score(test_labels, predicted_labels)*100:.2f}%')

