import numpy as np
from keras.layers import *
from imageio import imread
from math import sqrt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from keras.preprocessing import image
# from scipy.misc import imsave
import  numpy  as  np
from keras.layers import *
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dropout, Flatten, Dense
# from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import model_from_json
import os,cv2,pickle
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(240, 320, 3), padding='VALID'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='VALID'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(AveragePooling2D(pool_size=(19, 19)))

# set of FC => RELU layers
model.add(Flatten())
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.save_weights('models/model_weights.h5')
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()    
f = open('models/history.pckl', 'wb')
pickle.dump(model.history, f)
f.close()