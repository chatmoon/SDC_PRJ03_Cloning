# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:55:04 2017
@author: mo
"""
# In[ 0 ]: PREREQUISITES
# Import libraries
import csv
import cv2
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
import time
from tools import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas
import random
plt.style.use('ggplot')

# Parameter
nb_epoch     = 5  # 10
batch_size   = 32 # 32 50 1000
delta        = 0.25
input_shape  = (image_height, image_width, 3) # (160, 320, 3)

# In[ 1 ]: LOAD IMAGES AND LABELS
 
# Read driving_log.csv file
#import csv
row_names   = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
driving_log = pandas.read_csv(pathData2+'driving_log.csv', skiprows=[0], names=row_names)
center      = driving_log.center.tolist()
left        = driving_log.left.tolist()
right       = driving_log.right.tolist()
steering    = driving_log.steering.tolist()

# Split dataset into two set: train, validation
#from sklearn.model_selection import train_test_split
train_center  , valid_center   = train_test_split(center, test_size=0.2) # Do we need it?, see model.fit(xx, valid_split=0.2,xxx)
train_left    , valid_left     = train_test_split(left, test_size=0.2) 
train_right   , valid_right    = train_test_split(right, test_size=0.2) 
train_steering, valid_steering = train_test_split(steering, test_size=0.2) 

# In[ X ]: BUILD MODEL TO PREDICT MY STEERING ANGLE
# Generate training and validation dataset
train_generator = generator(train_center, train_left, train_right, train_steering, pathData3, batch_size, delta, image_width, image_height)
valid_generator = generator(valid_center, valid_left, valid_right, valid_steering, pathData3, batch_size, delta, image_width, image_height)

# Build the model: nvidea model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=input_shape))
model.add(Conv2D(24, (5, 5), padding='valid', activation='elu', strides=(2, 2)))
model.add(Conv2D(32, (5, 5), padding='valid', activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='valid', activation='elu', strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

# Compile the model
#import keras.backend as K
#adam = keras.optimizers.Adam(lr=0.000085, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
#K.set_value(adam.lr, 0.5 * K.get_value(sgd.lr))
model.compile(loss='mse', optimizer='adam')

# Callbacks.Checkpoint: fault tolerance technique
#from datetime import datetime as dt
#import time
postfix    = dt.now().strftime("%y%m%d_%H%M")
pathFile0  = pathData6+postfix+'/'  
pathFile1  = pathFile0+'ckpt_W_{epoch:02d}_{val_loss:.2f}.hdf5'

checkpoint = ModelCheckpoint(pathFile1, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop  = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
tensorboard= TensorBoard(log_dir=pathFile0, histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None) 
callbacks_list = [earlystop,checkpoint, tensorboard]

# Callbacks.History: display training_loss and val_loss
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
history    = model.fit_generator(train_generator,
                    steps_per_epoch  = len(train_steering),
                    epochs           = nb_epoch,
                    verbose          = 1,
                    callbacks        = callbacks_list,
                    validation_data       = valid_generator,
                    validation_steps = len(valid_steering)
                    ) 

# list history.keys()
print(history.history.keys())

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid.'], loc='upper left')
plt.show()

# Save the trained model
model.save_weights('model_weights.h5')
model.save(pathFile0+'model.h5')

'''
Next: download the model and see how well it drives the car in the simulator
'''