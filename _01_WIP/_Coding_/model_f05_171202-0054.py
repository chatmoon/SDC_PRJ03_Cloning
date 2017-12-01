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

# Parameter
nb_epoch     = 4  # 10
batch_size   = 32 # 32 50 1000
delta        = 0.2
input_shape  = (160, 320, 3)

# In[ 1 ]: LOAD IMAGES AND LABELS
 
# Read driving_log.csv file
#import csv
pathData0 = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
pathData1 = pathData0+'data/'
pathData2 = pathData1+'myData_171126-1643/' # 'sample/'  'myData_171126-1643/'
pathData3 = pathData2+'IMG/' # '../data/IMG' # <- to be updated with the AWS or Google path repo


lines = []
with open(pathData2+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Split dataset into two set: train, validation
#from sklearn.model_selection import train_test_split
train_lines, validation_lines = train_test_split(lines, test_size=0.2) # Do we need it?, see model.fit(xx, validation_split=0.2,xxx)

def generator(lines, batch_size=batch_size, delta=delta):
    num_lines = len(lines)
    while 1: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            # Import images and labels
            images = []
            angles = []
         
            for batch_sample in batch_lines: # for line in lines:
                for i in range(3):
                    # Update the localpath with the cloudpath
                    source_path = batch_sample[i] # source_path = line[i] # local path 
                    filename = source_path.split('\\')[-1] # ('/')[-1] # file name
                    current_path = pathData3 + filename

                    # Import images
                    image = cv2.imread(current_path)
                    images.append(image)
                    # Import output labels (steering angles)
                    angle = float(line[3])
                    if   i == 1: # images from the left
                        angles.append(angle+delta)
                    elif i == 2: # images from the right
                        angles.append(angle-delta)
                    else: # images from the center
                        angles.append(angle)

            # Convert images and labels into np.array
            X_train = np.array(images)
            y_train = np.array(angles)


            # In[ X ]: DATA AUGMENTATION
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))  # <- check if it should be 1
                augmented_angles.append(angle*(-1.0))

            # Convert images and labels into np.array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)

class save_w(Callback):
    """
    save model weights at the end of each epoch.
    """
    def __init__(self,path,modelname):
        self.path      = path
        self.modelname = modelname

    def on_epoch_end(self, epoch):
        self.model.save_weights(self.path+self.modelname+'_epoch_{}.h5'.format(epoch + 1))


# In[ X ]: BUILD MODEL TO PREDICT MY STEERING ANGLE
# Generate training and validation dataset
train_generator      = generator(train_lines, batch_size=batch_size, delta=delta)
validation_generator = generator(validation_lines, batch_size=batch_size, delta=delta)

# Build the model: nvidea model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=input_shape)) #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=X_train.shape[1:]))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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
model.compile(loss='mse', optimizer='adam')


model.fit_generator(train_generator,
                    steps_per_epoch  = len(train_lines),
                    epochs           = nb_epoch,
                    verbose          = 1,
                    callbacks        = [save_w(path=pathData0, modelname='nvidia')],  # [model.save_weights('my_model_weights.h5')]  # None, 
                    validation_data  = validation_generator,
                    validation_steps = len(validation_lines),
                    initial_epoch    = 0 ) 


## Spplit and shuffle dataset then Train the model
#        model.fit(X_train, y_train,/
#                  validation_split=0.2,/
#                  shuffle=True,/
#                  epochs=nb_epoch) 

# Save the trained model
model.save('model.h5')

'''
Next: download the model and see how well it drives the car in the simulator
'''