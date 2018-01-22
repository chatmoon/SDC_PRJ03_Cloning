import cv2 #, os
import numpy as np
#import matplotlib.image as mpimg
from sklearn.utils import shuffle
import argparse
# from newModel import parse_args
# Parameter
default_dir     = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
default_dataset	= 'sample' # 'myDebug' # 'sample'
# ex: dir_csv   = default_dir + 'data/' + default_dataset + '/'
# ex: dir_image = dir_csv + 'IMG/'
default_input_shape = (32, 155, 3) # (image_height, image_width, 3)


# Helper function(s): command-line / parse parameters
def parse_args():
	parser = argparse.ArgumentParser(prog='behavioral cloning', description='train or fine tune the model')
	parser.add_argument('-p', '--dir', help='root directory path', dest='dir', action='store', type=str, default=default_dir)
	parser.add_argument('-d', '--dtset', help='data set folder name', dest='dtset', action='store', type=str, default=default_dataset)
	parser.add_argument('-s', '--shape', help='tuple = ( image_height, image_width, chanel )', dest='input_shape', type=tuple, default=default_input_shape) # (32, 155, 3), (32, 32, 3) or (64, 64, 3)
	parser.add_argument('-e', '--epoch', help='number of epoch', dest='nb_epoch', type=int, default=10)
	parser.add_argument('-b', '--batch', help='batch size', dest='batch_size', type=int, default=32)
	parser.add_argument('-a', '--delta', help='delta angle', dest='delta', type=float, default=0.2)
	parser.add_argument('-v', '--show', help='plot and show the LOSS and VAL_LOSS graph', dest='show', action='store_true', default=False)
	parser.add_argument('-t', '--tune', help='activate the fine tune mode - by default it is in training mode', dest='tune', action='store_true', default=False)
	parser.add_argument('-f', '--freeze', help='freeze all layers except the last one when the fine tune mode is activated - by default these layers are unfrozen', dest='freeze', action='store_false')
	args   = parser.parse_args()

	return args

flags = parse_args()

def generator(flags, lines):
    num_lines = len(lines)
    dir_image = flags.dir+'data/'+flags.dtset+'/IMG/'
    while True: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_lines, flags.batch_size):
            batch_lines = lines[offset:offset+flags.batch_size]

            # Import images and labels
            images = []
            angles = []
         
            for batch_sample in batch_lines: # for line in lines:
                for i in range(3):
                    # Update the localpath with the cloudpath
                    source_path = batch_sample[i] # source_path = line[i] # local path 
                    filename = source_path.split('\\')[-1] # ('/')[-1] # file name
                    current_path = dir_image + filename
                    # Import images
                    image = cv2.imread(current_path)
                    images.append(image)
                    # Import output labels (steering angles)
                    angle = float(batch_sample[3])
                    if   i == 1: # images from the left
                        angles.append(angle+flags.delta)
                    elif i == 2: # images from the right
                        angles.append(angle-flags.delta)
                    else: # images from the center
                        angles.append(angle)

            # In[ X ]: DATA AUGMENTATION
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images,angles):
                augmented_image, augmented_angle = [], []
                # Crop
                augmented_image = image[70:-25, :, :]    # (65,320,3)
                # Resize: http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
                augmented_image = cv2.resize(augmented_image, (flags.input_shape[1], flags.input_shape[0]), cv2.INTER_AREA)
                # Save before randomly flipt or changing brightness
                augmented_images.append(augmented_image) # (32,155,3)
                augmented_angles.append(angle)
                # Randomly flipt the image and adjust the steering angle
                augmented_image = cv2.flip(augmented_image,1)
                augmented_angle = angle*(-1.0)
                if np.random.rand() < 0.5:
                    # Randomly change brightness (to simulate day and night conditions)
                    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2HSV) # hsv: hue, saturation, value
                    rate            = 1.0 + 0.4 * (np.random.rand() - 0.5)
                    augmented_image[:,:,2] =  augmented_image[:,:,2] * rate
                    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_HSV2RGB)
                    # Add to the list
                    augmented_images.append(augmented_image)
                    augmented_angles.append(augmented_angle)

            # Convert images and labels into np.array
            X_train = np.array(augmented_images)  # X_train.shape = (143, 32, 155, 3)
            y_train = np.array(augmented_angles)


            yield shuffle(X_train, y_train)