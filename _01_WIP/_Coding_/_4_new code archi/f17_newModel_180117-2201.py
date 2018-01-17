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
from newTool import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

# Parameter
# image_height, image_width =  160, 320 # DELETE
# path_defaut = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
# path_
# path_data   = path_defaut+'data/'+'sample/'
dir_defaut = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/data/'
data_set	= 'sample'
# ex: dir_csv   = dir_defaut + data_set + '/'
# ex: dir_image = dir_csv + 'IMG/'

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-dir', help='directory path', dest='dir', type=str, default=dir_defaut)
	parser.add_argument('-dtset', help='data set folder name', dest='dtset', type=str, default=data_set)
	parser.add_argument('-shape', help='tuple: image_height, image_width, chanel', dest='input_shape', type=tuple, default=(image_height, image_width, 3)) # -> pas bonne valeur : (160, 320, 3)
	parser.add_argument('-epoch', help='number of epoch', dest='nb_epoch', type=int, default=10)
	parser.add_argument('-batch', help='batch size', dest='batch_size', type=int, default=32)
	parser.add_argument('-delta', help='delta angle', dest='delta', type=float, default=0.2)
	parser.add_argument('-show', help='boolean: (not) show the LOSS and VAL_LOSS figure', dest='show', type=bool, default=True)
	args   = parser.parse_args()

	return args

# In[ 1 ]: LOAD IMAGES AND LABELS
def data_load(args):
	# Read driving_log.csv file
	lines = []
	pathData2 = args.dir+args.dtset+'/'
	with open(pathData2+'driving_log.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    for line in reader:
	        lines.append(line)

	# Split dataset into two set: train, validation
	train_lines, validation_lines = train_test_split(lines, test_size=0.2) # Do we need it?, see model.fit(xx, validation_split=0.2,xxx)

	return train_lines, validation_lines


def model_build(args):
	# Build the model: nvidea model
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0-0.5, input_shape=args.input_shape))
	#model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Conv2D(16, (5, 5), activation='elu', strides=(2, 2)))
	model.add(Conv2D(32, (5, 5), activation='elu', strides=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='elu', strides=(2, 2)))
	#model.add(Conv2D(64, (3, 3), activation='elu'))
	#model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	model.summary()

	return model


def model_train(args, model, train_lines, validation_lines):
	# Generate training and validation dataset
	train_generator      = generator(train_lines, batch_size=args.batch_size, delta=args.delta, image_width=args.input_shape[1], image_height=args.input_shape[0])
	validation_generator = generator(validation_lines, batch_size=args.batch_size, delta=args.delta, image_width=args.input_shape[1], image_height=args.input_shape[0])

	# Compile the model
	model.compile(loss='mse', optimizer='adam')

	# Callbacks.Checkpoint: fault tolerance technique
	##from datetime import datetime as dt
	##import time
	postfix    = dt.now().strftime("%y%m%d_%H%M")
	pathFile0  = args.dir+'logs/nn_logs/'+postfix+'/'  
	pathFile1  = pathFile0+'ckpt_W_{epoch:02d}_{val_loss:.2f}.hdf5'

	checkpoint = ModelCheckpoint(pathFile1, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlystop  = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
	tensorboard= TensorBoard(log_dir=pathFile0, histogram_freq=0, batch_size=args.batch_size, write_graph=False, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None) 
	callbacks_list = [earlystop,checkpoint, tensorboard]
	
	# Callbacks.History: display training_loss and val_loss
	##import matplotlib.pyplot as plt
	##import matplotlib.image as mpimg
	history    = model.fit_generator(train_generator,
	                    steps_per_epoch  = len(train_lines),
	                    epochs           = args.nb_epoch,
	                    verbose          = 1,
	                    callbacks        = callbacks_list,
	                    validation_data  = validation_generator,
	                    validation_steps = len(validation_lines)
	                    ) 

	# list history.keys()
	#print(history.history.keys())

	# plot loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'valid.'], loc='upper left')
	plt.savefig(pathFile0+'figure_'+postfix+'.png')
	if args.show:
		plt.show()

	# Save the trained model
	model.save_weights(pathFile0+'model_weights.h5')
	model.save(pathFile0+'model.h5')

	return history


def main():
	args = parse_args()
	image_height, image_width = args.input_shape[0], args.input_shape[1]
	# Load and split data set
	train_lines, validation_lines = data_load(args)
	# Build a model
	model = model_build(args)
	# Train and save the model
	history = model_train(args, model, train_lines, validation_lines)


if __name__ == '__main__':
    main()