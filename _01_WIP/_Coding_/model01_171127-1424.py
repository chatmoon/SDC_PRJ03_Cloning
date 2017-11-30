# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:55:04 2017
@author: mo
sandbox: model00 is a restranscription of the episode 7 (training the network) in lesson 12 (prj Behavioral cloning)
"""
# Import libraries
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# In[ 1 ]: LOAD IMAGES AND LABELS
 
# Read driving_log.csv file
#import csv
pathData = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/data/myData_171126-1643/'
pathCloud = pathData+'IMG/' # '../data/IMG' # <- to be updated with the AWS or Google path repo

lines = []
with open(pathData+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Import images and labels
#import cv2
images = []
measurements = []

for line in lines:
    # Update the localpath with the cloudpath
    source_path = line[0] # local path
    filename = source_path.split('\\')[-1] # ('/')[-1] # file name
    current_path = pathCloud + filename
    # Import images
    image = cv2.imread(current_path)
    images.append(image)
    # Import output labels (steering measurements)
    measurement = float(line[3])
    measurements.append(measurement)

# Convert images and labels into np.array
#import numpy as np
X_train = np.array(images)
y_train = np.array(measurements)


# In[ 2 ]: BUILD MODEL TO PREDICT MY STEERING ANGLE
'''
The model should be a regression network: prediction of the steering angle.
For a classification network, we used a softmax activation function to the output layer.
But in a regression network, we want the single output node to directly predit the steering measurement.
Then we will not apply an activation function here.

For the loss function, we will use mean square error or MSE.
It is different than the cross-entropy function.
Because it is a regression network instead of a classification network.

The goal is to minimize the error between the steering measurement
that the network predicts and the ground truth steering measurement.
And MEAN SQUARED ERROR is a good loss function for this
''' 
#from keras.models import Sequential
#from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint

# Build the model
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=X_train.shape[1:]))
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

# Save the trained model
model.save('model.h5')

'''
Next: download the model and see how well it drives the car in the simulator
'''