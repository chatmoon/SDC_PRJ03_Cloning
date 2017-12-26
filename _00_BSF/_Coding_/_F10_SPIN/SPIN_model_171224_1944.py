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
from tools import generator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameter
nb_epoch     = 5 # 30  # 10
batch_size   = 32 # 32 50 1000
delta        = 0.25 # 0.2
input_shape  = (image_height, image_width, 3) # (160, 320, 3)

# In[ 1 ]: LOAD IMAGES AND LABELS
 
# Read driving_log.csv file
#import csv
lines = []
with open(pathData2+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Split dataset into two set: train, validation
#from sklearn.model_selection import train_test_split
train_lines, validation_lines = train_test_split(lines, test_size=0.2) # Do we need it?, see model.fit(xx, validation_split=0.2,xxx)


# In[ X ]: BUILD MODEL TO PREDICT MY STEERING ANGLE
# Generate training and validation dataset
train_generator      = generator(train_lines, batch_size=batch_size, delta=delta, image_width=image_width,image_height=image_height)
validation_generator = generator(validation_lines, batch_size=batch_size, delta=delta, image_width=image_width,image_height=image_height)

# Build the model: nvidea model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=input_shape))
model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

# Compile the model
adam = Adam(lr=1.0e-4) # 0.0001)
model.compile(loss='mse', optimizer=adam) # 'adam')

# Callbacks.Checkpoint: fault tolerance technique
#from datetime import datetime as dt
#import time
postfix    = dt.now().strftime("%y%m%d_%H%M")
pathFile0  = pathData6+postfix+'/'  
pathFile1  = pathFile0+'ckpt_W_{epoch:02d}_{val_loss:.2f}.hdf5'

checkpoint = ModelCheckpoint(pathFile1, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop  = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
tensorboard= TensorBoard(log_dir=pathFile0, histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None) 
callbacks_list = [earlystop,checkpoint, tensorboard]

# Callbacks.History: display training_loss and val_loss
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
history    = model.fit_generator(train_generator,
                    steps_per_epoch  = len(train_lines),
                    epochs           = nb_epoch,
                    verbose          = 1,
                    callbacks        = callbacks_list,
                    validation_data  = validation_generator,
                    validation_steps = len(validation_lines)
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
#model.save_weights('model_weights.h5')
model.save(pathFile0+'model.h5')

'''
Next: download the model and see how well it drives the car in the simulator
'''