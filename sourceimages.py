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
sess = tf.InteractiveSession()
print(glob.glob("/Users/shind/OneDrive/Second Semester/Analytical Data Mining/Project"))

source_path = 'Alopecia_areata_images'
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
#shape = [height, width, channels]
#x = tf.placeholder(dtype = tf.float32)
'''flip_2 = tf.image.flip_up_down(img)
flip_3 = tf.image.flip_left_right(img)
flip_4 = tf.image.random_flip_up_down(img)
flip_5 = tf.image.random_flip_left_right(img)'''

imgplot = plt.imshow(img1)
#plt.savefig('new.jpg')


#With Tensorflow:

img =plt.imread(images[1])
tf_img = tf.convert_to_tensor(img)
flip1_img = tf.image.flip_left_right(tf_img) #flip
flip2_img = tf.image.flip_up_down(tf_img) #flip
brght_img1 = tf.image.rot90(tf_img, k = 1) #rotate
brght_img2 = tf.image.rot90(tf_img, k = 2) #rotate
brght_img3 = tf.image.rot90(tf_img, k = 3) #rotate
brght_img4 = tf.image.rot90(tf_img, k = 4) #rotate
brght_img = tf.image.adjust_brightness(tf_img, delta= 0.2) #brightness


img_dimension = tf.shape(img).eval()
height,width,channel=img_dimension
original_size = [height, width, channel]
x = tf.placeholder(dtype = tf.float32, shape = original_size)
noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0,
dtype=tf.float32) #noise
output = tf.add(x, noise) #addnoise
