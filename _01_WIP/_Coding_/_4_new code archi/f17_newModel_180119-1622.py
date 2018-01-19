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
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.initializers import TruncatedNormal, Constant
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
import time
from newTool import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.image as mpimg
import pandas
import random
import argparse

# Parameter
dir_defaut = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
data_set	= 'sample'
# ex: dir_csv   = dir_defaut + 'data/' + data_set + '/'
# ex: dir_image = dir_csv + 'IMG/'

# Helper function(s): create directory tree
def dir_check(path):
    '''Create a folder if not present'''
    if not os.path.exists(path):
        os.makedirs(path)

def dir_create(path, dir_dictionary):
    dir_check(path)

    for dir_root in dir_dictionary['root']:                # dir_root      = logs
        dir_check(path+dir_root+'/')            # args.dir/logs
        try:
            for dir_subroot in dir_dictionary[ dir_root ]: # dir_subroot   = nn_logs
                dir_check(path+dir_root+'/'+dir_subroot+'/')
        except:
            pass

# Helper function(s): command-line / parse parameters
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-dir', help='directory path', dest='dir', type=str, default=dir_defaut)
	parser.add_argument('-dtset', help='data set folder name', dest='dtset', type=str, default=data_set)
	parser.add_argument('-shape', help='tuple: image_height, image_width, chanel', dest='input_shape', type=tuple, default=(image_height, image_width, 3)) # -> pas bonne valeur : (160, 320, 3)
	parser.add_argument('-epoch', help='number of epoch', dest='nb_epoch', type=int, default=10)
	parser.add_argument('-batch', help='batch size', dest='batch_size', type=int, default=32)
	parser.add_argument('-delta', help='delta angle', dest='delta', type=float, default=0.2)
	parser.add_argument('-show', help='show the LOSS and VAL_LOSS graph', dest='show', action='store_true')
	parser.add_argument('-tune', help='activate the fine tune mode - by default the training mode is activated', dest='tune', action='store_true')
	args   = parser.parse_args()

	return args

# Helper function(s): load lines from the driving_log.csv file
def data_load(args):
	# Read driving_log.csv file
	lines = []
	pathData2 = args.dir+'data/'+args.dtset+'/'
	with open(pathData2+'driving_log.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    for line in reader:
	        lines.append(line)

	# Split dataset into two set: train, validation
	train_lines, validation_lines = train_test_split(lines, test_size=0.2) # Do we need it?, see model.fit(xx, validation_split=0.2,xxx)

	return train_lines, validation_lines

# Helper function(s): build a model similar to nvidea CNN
def model_build(args):
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

# Helper function(s): callbacks_list, a fault tolerance technique
def callback_function(args):
	postfix  = dt.now().strftime("%y%m%d_%H%M")
	dir_log  = args.dir+'logs/nn_logs/'+postfix+'/'  
	dir_ckpt = dir_log+'ckpt_W_{epoch:02d}_{val_loss:.2f}.hdf5'

	checkpoint = ModelCheckpoint(dir_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlystop  = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
	tensorboard= TensorBoard(log_dir=dir_log, histogram_freq=0, batch_size=args.batch_size, write_graph=False, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None) 
	callbacks_list = [earlystop,checkpoint, tensorboard]

	return callbacks_list, dir_log

# Helper function(s): plot loss and validation_loss
def plot_loss(args, history, dir_log):
	# plot loss and validation_loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'valid.'], loc='upper left')
	plt.savefig(dir_log+'figure.png')

	if args.show:
		plt.show()

# Helper function(s): fine tune the model
def model_tune(args, dir_model_h5, name_model_h5='model.h5'):
	# load the pre-trained model
	model = load_model(dir_model_h5+name_model_h5)
	# slice off the end of the neuural network
	model.pop() 
	# add a new fully connected layer (doc: https://keras.io/initializers/)
	model.add(Dense(1, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1, seed=None),
                       bias_initializer=Constant(value=0.1)))
	model.layers[-1].name = 'dense_4'

	# freeze all layers except the last one
	for i, layer in enumerate(model.layers[:-1]):
	    layer.trainable = True # False
	    # note: finally, I unfreezed all layers and I got a better result at the end
	    #print('< {} > {}'.format(i, layer.name))
	# unfreeze the last layer
	model.layers[-1].trainable = True

	# Compile the model
	#adam = Adam(lr=8.5e-4) #, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
	#model.compile(loss='mse', optimizer=adam) # 'adam')

	return model

# Helper function(s): train the model
def model_train(args, model, train_lines, validation_lines):
	# Generate training and validation dataset
	train_generator      = generator(train_lines, batch_size=args.batch_size, delta=args.delta, image_width=args.input_shape[1], image_height=args.input_shape[0])
	validation_generator = generator(validation_lines, batch_size=args.batch_size, delta=args.delta, image_width=args.input_shape[1], image_height=args.input_shape[0])

	callbacks_list, dir_log = callback_function(args)

	if args.tune:
		# note: if <python model.py -tune> then <fine tune>
		print()
		print('|>>>>| mode: {}     |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|'.format('tune'))
		print()
		model = model_tune(args, args.dir+'data/h5/', 'model.h5')

		# Compile the model
		adam = Adam(lr=8.5e-4) #, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
		model.compile(loss='mse', optimizer=adam) # 'adam')
	else:
		# note: if <python model.py>       then <train>
		print()
		print('|>>>>| mode: {} |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|'.format('training'))
		print()
		# Compile the model
		model.compile(loss='mse', optimizer='adam')
	
	# Train/Tune the model
	history    = model.fit_generator(train_generator,
	                    steps_per_epoch  = len(train_lines),
	                    epochs           = args.nb_epoch,
	                    verbose          = 1,
	                    callbacks        = callbacks_list,
	                    validation_data  = validation_generator,
	                    validation_steps = len(validation_lines)
	                    ) 

	# Save the trained model
	model.save_weights(args.dir+'data/h5/'+'model_weights.h5')
	model.save(args.dir+'data/h5/'+'model.h5')

	# Save a copy in the archive/log
	model.save_weights(dir_log+'model_weights.h5')
	model.save(dir_log+'model.h5')
	
	# plot loss and validation_loss
	plot_loss(args, history, dir_log)

	return history


def main():
	# parse a set of parameters
	args = parse_args()

	# parameters
	image_height, image_width = args.input_shape[0], args.input_shape[1]

	# create directory tree
	dir_dict = {}
	dir_dict = {'root' : ['logs', 'data'],\
    			'data' : ['h5'],\
    			'logs' : ['nn_logs'] }
	dir_create(args.dir, dir_dict)

	# Load and split data set
	train_lines, validation_lines = data_load(args)

	# Build a model
	model = model_build(args)

	# Train and save the model
	history = model_train(args, model, train_lines, validation_lines)


if __name__ == '__main__':
    main()