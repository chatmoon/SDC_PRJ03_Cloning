import cv2, os
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle

image_height, image_width =  32, 155
pathData0 = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
pathData1 = pathData0+'data/'
pathData2 = pathData1+'myDebug/' # BEFORE USING SAMPLE CHANGE DRIVE.PY ',' vs '.' # 'myDebug/' 'sample/' < NOT USE 'myData_171202-0037/' >
pathData3 = pathData2+'IMG/' # '../data/IMG' # <- to be updated with the AWS or Google path repo
pathData4 = pathData0+'logs/'
pathData5 = pathData4+'model/'
pathData6 = pathData4+'nn_logs/' 

def generator(lines, batch_size=32, delta=0.2, image_width=image_width,image_height=image_height):
    num_lines = len(lines)
    while True: # Loop forever so the generator never terminates
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
                    angle = float(batch_sample[3])
                    if   i == 1: # images from the left
                        angles.append(angle+delta)
                    elif i == 2: # images from the right
                        angles.append(angle-delta)
                    else: # images from the center
                        angles.append(angle)

            # In[ X ]: DATA AUGMENTATION
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images,angles):
                # augmented_image, augmented_image, augmented_image, augmented_angle, augmented_image = [], [], [], [], []
                augmented_image, augmented_angle = [], []
                # Crop
                augmented_image = image[70:-25, :, :]    # (65,320,3)
                # Resize: http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
                augmented_image = cv2.resize(augmented_image, (image_width, image_height), cv2.INTER_AREA)
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