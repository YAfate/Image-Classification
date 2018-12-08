# Image-Classification
Image classification for skin diseases.

Data set : images of skin diseases.
main source : https://www.dermnetnz.org

CNN  for image classification               
To run, change the folder of training data and the testing image data.
Image classification of two classes. 
Uses Tensorflow package of python.
May give the warning " I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2"  but the code works fine.

KNN for image classification:
The variable 'k' holds the number of neighbours to be selected.
The label of a image is extracted from the image name, for example, if you want an image of a tree to be classified as a tree, the name in thetraining model should be tree12.jpg.
The 'imagePaths' varible holds the source folder paths.

Data Augmentation

The augmentation factor : 360
The code rotates each image from 1 degrees to 180 degrees (included) and then flips each rotated image horizontally.


To run the code : Change the source path to where the original images are stored.

To change the augmentation factor : Change the variable 'i' which represents the degree of rotation.


Libraries Used :
glob
os
numpy
scipy , ndarray
skimage
skimage.util
matplotlib.pyplot
tensorflow
math
PIL
sklearn
imutils
cv2
re
keras
Sequential ,  Dense, Conv2D, MaxPooling2D, Flatten from Keras
