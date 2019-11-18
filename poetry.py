from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import cv2 as cv

import numpy
from PIL import Image

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.load_weights('./Shelf_classifier.h5')


import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
def predictor(img):
  # img = image.load_img(link,target_size=(64,64, 3))
  width=64
  height=64
  img = cv.cvtColor(numpy.array(img), cv.COLOR_BGR2RGB)
  img = cv.resize(img,(width,height))
  print(type(img))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = img/255
  i=classifier.predict(np.array(img))
  if(i>0.5):
    return ("The Shelf is filled ..... probability that shelf is filled : {} \n".format(i))
  else:
    return ("The Shelf is Empty ...... probability that shelf is filled : {} \n".format(i))

# predictor('a.png')