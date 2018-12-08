import glob
import os
import numpy
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import math
from PIL import Image

sess = tf.InteractiveSession()
print(glob.glob("/Users/shind/OneDrive/Second Semester/Analytical Data Mining/Project")) #The path to the folder of images

source_path = 'Alopecia_areata_images' #Image folder
desired_copies = 1000

images = [os.path.join(source_path, f) for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]

new_copies = 0

for image in images:
    image_path = image
    image_data = sk.io.imread(image_path)
    
'''img = mpimg.imread(images[1])
imgplot = plt.imshow(img)
plt.show()'''

img = mpimg.imread(images[1])
img1 = np.fliplr(img)   # To flip the image horizontally. Augmentation factor = 2x


#With Tensorflow:

'''img =plt.imread(images[1])
tf_img = tf.convert_to_tensor(img)
flip1_img = tf.image.flip_left_right(tf_img) #flip '''

#brght_img = tf.image.adjust_brightness(tf_img, delta= 0.2) #brightness change


img_dimension = tf.shape(img).eval()
height,width,channel=img_dimension
original_size = [height, width, channel]
x = tf.placeholder(dtype = tf.float32, shape = original_size)

for image in images:
    for i in range(1,181): 
        new_image = tf.image.decode_jpeg(tf.read_file(images[1]), channels=3)
        new_imager = tf.contrib.image.rotate(new_image, math.radians(i))
        new_imagef = tf.image.flip_left_right(new_imager)
        new_imager = tf.image.encode_jpeg(new_imager)
        new_imagef = tf.image.encode_jpeg(new_imagef)
        with tf.Session():
            tf.write_file("rotate"+str(i)+".jpeg", new_imager).run() # augmentation factor : 180
            tf.write_file("flip"+str(i)+".jpeg", new_imagef).run() # augmentation factor : 180



 
sess.close()
