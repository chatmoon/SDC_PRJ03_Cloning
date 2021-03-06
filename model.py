# -*- coding: utf-8 -*-
"""
Created on Mon Sat 23 11:55:04 2017
@author: mo

#ENVIRONMENT :
- platform : win10-64
- GPU: Nvidia Geforce 840M
- tensorflow-gpu 1.1.0 <pip>
- Keras 2.1.1 <pip>
- python 3.5.2 0
- numpy 1.12.1 <pip>
- opencv3 3.1.0 py35_0 menpo
- scikit-learn 0.19.0 np112py35_0
- scipy 1.0.0 <pip>
- h5py 2.7.1 <pip>
- hdf5 1.10.1 h98b8871_1 anaconda
- eventlet 0.21.0 <pip>
- Flask 0.12.2 <pip>
- greenlet 0.4.12 <pip>
- h5py 2.7.1 <pip>
- hdf5 1.10.1 h98b8871_1 anaconda
- imageio 2.1.2 <pip>
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
from tools import generator, parse_args
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.image as mpimg
import pandas
import random
import argparse

# In[ 1 ]: HELPER FUNCTIONS
# Helper function(s): create directory tree
def dir_check(path):
    '''Create a folder if not present'''
    if not os.path.exists(path):
        os.makedirs(path)

def dir_create(path, dir_dictionary):
    dir_check(path)

    for dir_root in dir_dictionary['root']:               	# dir_root      = logs
        dir_check(path+dir_root+'/')            			# args.dir/logs
        try:
            for dir_subroot in dir_dictionary[ dir_root ]:	# dir_subroot   = nn_logs
                dir_check(path+dir_root+'/'+dir_subroot+'/')
        except:
            pass

# Helper function(s): load lines from the driving_log.csv file
def data_load(args):
	# Read driving_log.csv file
	lines = []
	dir_csv = args.dir+'data/'+args.dtset+'/'
	with open(dir_csv+'driving_log.csv') as csvfile:
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

	# freeze or not all layers except the last one
	# note: if < python model.py -tune -freeze > then args.freeze = False else True
	if args.freeze: 
		print()
		print('|>>>>|     model.layers[:-1] are trainable - args.freeze: {}     |>>>>|'.format(args.freeze))
		print()
	else:
		print()
		print('|>>>>|     model.layers[:-1] are frozen - args.freeze: {}     |>>>>|'.format(args.freeze))
		print()

	for i, layer in enumerate(model.layers[:-1]):
	    layer.trainable = args.freeze 
	    # note: finally, I unfreezed all layers and I got a better result at the end
	    # print('< {} > {}'.format(i, layer.name))
	# unfreeze the last layer
	model.layers[-1].trainable = True

	return model

# Helper function(s): train the model
def model_train(args, model, train_lines, validation_lines):
	# Generate training and validation dataset
	train_generator      = generator(args, train_lines) # batch_size=args.batch_size, delta=args.delta, image_width=args.input_shape[1], image_height=args.input_shape[0])
	validation_generator = generator(args, validation_lines) #batch_size=args.batch_size, delta=args.delta, image_width=args.input_shape[1], image_height=args.input_shape[0])

	callbacks_list, dir_log = callback_function(args)

	if args.tune:
		# note: if <python model.py -tune> then <fine tune>
		print()
		print('|>>>>|     mode: {}     |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|'.format('tune'))
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


# In[ 2 ]: TOP-LEVEL SCRIPT
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