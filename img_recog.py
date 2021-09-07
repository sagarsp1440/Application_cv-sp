#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans


# In[57]:


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


# In[58]:


import os
foods = ['Cakes', 'Pasta', 'Pizza']
path = 'C:/Users/sagar/Documents/APP of CV&SP/Ontrack resources/Resources_4.1/FoodImages/'
training_file_names = []
training_food_labels = []
for i in range(0, len(foods)):
    sub_path = path + 'Train/' + foods[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_food_labels = [i] * len(sub_file_names) #create a list of N elements, all are i
    training_file_names += sub_file_names
    training_food_labels += sub_food_labels
    


# In[59]:


num_words = 50
dictionary_name = 'food'
diction = Dictionary(dictionary_name, training_file_names, 50)


# In[60]:


training_word_hist = diction.learn()


# In[61]:


import pickle
#save dictionary
with open('food_dictionary.dic', 'wb') as f: #'wb' is for binary write
    pickle.dump(diction, f)


# In[62]:


import pickle #you may not need to import it if this has been done
with open('food_dictionary.dic', 'rb') as f: #'rb' is for binary read
    diction = pickle.load(f)


# In[63]:


num_nearest_neighbours = 25 #number of neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours)
knn.fit(training_word_hist, training_food_labels)


# In[64]:


test_file_names = ['C:/Users/sagar/Documents/APP of CV&SP/Ontrack resources/Resources_4.1/FoodImages/Test/Pasta/pasta35.jpg']
word_histograms = diction.create_word_histograms(test_file_names)
predicted_food_labels = knn.predict(word_histograms)
print('Food label: ', predicted_food_labels)


# In[65]:


from sklearn.metrics import accuracy_score, confusion_matrix
test_file_names = []
test_food_lables = []
for i in range(0,len(foods)):
    sub_path = path + 'Test/' + foods[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_food_labels = [i] * len(sub_file_names)
    test_file_names= sub_file_names
    test_food_lables = sub_food_labels
    word_histograms = diction.create_word_histograms(test_file_names)
    predicted_food_labels = knn.predict(word_histograms)
    print('Test label: ', test_food_lables)
    print('Food label: ', predicted_food_labels)


# In[66]:


ks = range(1,5)
for k in ks:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(training_word_hist, training_food_labels)
    predicted_food_labels = knn.predict(word_histograms)
    print(f"""accuracy for k={k}: {accuracy_score(test_food_lables, predicted_food_labels)*100:.2f}%
confusion matrix:
{confusion_matrix(test_food_lables, predicted_food_labels)}
--------------------------------------------
""")
    


# In[67]:


import sklearn.metrics


# In[68]:


nn = [5,10,15,20,25]
acc=[]
for n in nn:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(training_word_hist, training_food_labels)
    predicted_food_labels= knn.predict(word_histograms)
    acc.append(sklearn.metrics.accuracy_score(test_food_lables,predicted_food_labels))
    
    
print(acc)    


# In[69]:


from sklearn import svm
svm_classifier = svm.SVC(C = 50, #see slide 32 in week 4 lecture slides
                         kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_hist, training_food_labels)


# In[70]:


from sklearn import svm
svm_classifier = svm.SVC(C = 50, #see slide 32 in week 4 lecture slides
kernel = 'linear') #see slide 35 in week 4 lecture slides
svm_classifier.fit(training_word_hist, training_food_labels)


# In[71]:


test_file_names = ['C:/Users/sagar/Documents/APP of CV&SP/Ontrack resources/Resources_4.1/FoodImages/Test/Pasta/pasta35.jpg']
word_histograms = diction.create_word_histograms(test_file_names)
predicted_food_labels = svm_classifier.predict(word_histograms)
print('Food label: ', predicted_food_labels)


# In[72]:


test_file_names = []
test_food_lables = []
for i in range(0,len(foods)):
    sub_path = path + 'Test/' + foods[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_food_labels = [i] * len(sub_file_names)
    test_file_names= sub_file_names
    test_food_lables = sub_food_labels
    word_histograms = diction.create_word_histograms(test_file_names)
    predicted_food_labels = svm_classifier.predict(word_histograms)
    print('Test label: ', test_food_lables)
    print('Food label: ', predicted_food_labels)


# In[73]:


from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(test_food_lables, predicted_food_labels)
print(cm)


# In[74]:


c = [10,20,30,40,50]
acc=[]
for n in c:
    svm_classifier = svm.SVC(C = n, kernel = 'linear')
    svm_classifier.fit(training_word_hist, training_food_labels)
    predicted_food_labels= svm_classifier.predict(word_histograms)
    print(f"""accuracy for k: {accuracy_score(test_food_lables, predicted_food_labels)*100:.2f}%
confusion matrix:
{confusion_matrix(test_food_lables, predicted_food_labels)}
--------------------------------------------
""")

    
    
print(acc)    


# In[75]:


from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(n_estimators = 150, #weak classifiers
random_state = 0)
adb_classifier.fit(training_word_hist, training_food_labels)


# In[76]:


test_file_names = ['C:/Users/sagar/Documents/APP of CV&SP/Ontrack resources/Resources_4.1/FoodImages/Test/Pasta/pasta35.jpg']
word_histograms = diction.create_word_histograms(test_file_names)
predicted_food_labels = adb_classifier.predict(word_histograms)
print('Food label: ', predicted_food_labels)


# In[77]:


test_file_names = []
test_food_lables = []
for i in range(0,len(foods)):
    sub_path = path + 'Test/' + foods[i] + '/'
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
    sub_food_labels = [i] * len(sub_file_names)
    test_file_names= sub_file_names
    test_food_lables = sub_food_labels
    word_histograms = diction.create_word_histograms(test_file_names)
    predicted_food_labels = adb_classifier.predict(word_histograms)
    print('Food label: ', predicted_food_labels)


# In[79]:


a = [50,100,150,200,250]
acc=[]
for n in a:
    adb_classifier = AdaBoostClassifier(n_estimators = n,random_state = 0)
    adb_classifier.fit(training_word_hist, training_food_labels)
    predicted_food_labels= adb_classifier.predict(word_histograms)
    print(f"""accuracy for k: {accuracy_score(test_food_lables, predicted_food_labels)*100:.2f}%
confusion matrix:
{confusion_matrix(test_food_lables, predicted_food_labels)}
--------------------------------------------
""")

    
    
print(acc) 


# In[ ]:





# In[ ]:




