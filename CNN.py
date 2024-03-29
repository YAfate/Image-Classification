from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('cnndata',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('cnntest',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary') # 'cnndata' : the folder that contains training data, each class in different folder


classifier.fit_generator(training_set,
steps_per_epoch = 10,
epochs = 10,
validation_data = test_set,
validation_steps = 10)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('testimage16.jpg', target_size = (64, 64)) # name of the image to test. (in the same folder)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'Alopecia areata' # class 1 
else:
	prediction = 'Amelanotic Melanoma' #class 2
	
print("test image :")
print(prediction)
