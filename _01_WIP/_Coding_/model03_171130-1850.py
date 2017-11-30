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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Parameter
nb_epoch = 10

# In[ 1 ]: LOAD IMAGES AND LABELS
 
# Read driving_log.csv file
#import csv
pathData0 = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/data/'
pathData1 = pathData0+'myData_171126-1643/' #+'sample/' # 
pathData2 = pathData1+'IMG/' # '../data/IMG' # <- to be updated with the AWS or Google path repo

lines = []
with open(pathData1+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Import images and labels
#import cv2
images = []
measurements = []

for line in lines:
    for i in range(3): # << [MO] what about the steering correction
        # Update the localpath with the cloudpath
        source_path = line[i] # local path 
        filename = source_path.split('\\')[-1] # ('/')[-1] # file name
        current_path = pathData2 + filename
        # Import images
        image = cv2.imread(current_path)
        images.append(image)
        # Import output labels (steering measurements)
        measurement = float(line[3])
        if   i == 1: # images from the left
            measurements.append(measurement+0.2)
        elif i == 2: # images from the right
            measurements.append(measurement-0.2)
        else: # images from the center
            measurements.append(measurement)

# Convert images and labels into np.array
#import numpy as np
X_train = np.array(images)
y_train = np.array(measurements)

# In[ X ]: DATA AUGMENTATION
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))  # <- check if it should be 1
    augmented_measurements.append(measurement*(-1.0))

# Convert images and labels into np.array
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# In[ X ]: BUILD MODEL TO PREDICT MY STEERING ANGLE
#from keras.models import Sequential
#from keras.layers import Flatten, Dense, Lambda, Dropout
#from keras.layers.convolutional import Conv2D, Cropping2D
#from keras.layers.pooling import MaxPooling2D
#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Build the model: nvidea model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=X_train.shape[1:])) #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=X_train.shape[1:]))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Spplit and shuffle dataset then Train the model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=nb_epoch)

# Save the trained model
model.save('model.h5')

'''
Next: download the model and see how well it drives the car in the simulator
'''