from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import re


#The images are resized 32x32.
#And the image arrays are flattened using flatten function
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

# I have used 8 bins per channel and extracting a histogram coloured. 
def histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
 
	# normalizing step
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	else:
		cv2.normalize(hist, hist)
	#at place normalization
	return hist.flatten()

# grab the list of images that we'll be describing
imagePaths = list(paths.list_images("knndata"))


rawImages = []
features = []

#labels to hold classes
labels = []

for (i, imagePath) in enumerate(imagePaths):
	
	
	#label is extracted from image name
	# example : name of the image = "test1224.jpg", label = test
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[1]
	
	#print(label)
	
	pixels = image_to_feature_vector(image)
	hist = histogram(image)
	
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
 
	'''#get dupdates at intervals
	if i > 0 and i % 5 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))'''

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

	
# data partition
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.9, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.9, random_state=42)


	

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
k = 20
model = KNeighborsClassifier(n_neighbors=k,
	n_jobs=5)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("k = ",k)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))



# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=k,
	n_jobs=5)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("k = ",k)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))