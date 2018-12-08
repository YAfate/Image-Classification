# Image-Classification
Image classification for skin diseases.

Data set : images of skin diseases.
main source : https://www.dermnetnz.org

CNN  for image classification               
To run, change the folder of training data and the testing image data.
Image classification of two classes. 
Uses Tensorflow package of python.
May give the warning " I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2"  but the code works fine.

Data Augmentation

The augmentation factor : 360
The code rotates each image from 1 degrees to 180 degrees (included) and then flips each rotated image horizontally.
Libraries used :
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

To run the code : Change the source path to where the original images are stored.

To change the augmentation factor : Change the variable 'i' which represents the degree of rotation.
