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
plt.show()
