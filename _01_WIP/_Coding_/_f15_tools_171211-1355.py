import cv2, os
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import random

pathData0 = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
pathData1 = pathData0+'data/'
pathData2 = pathData1+'sample/' # 'sampleDirty/' 'myDebug/' 'sample/' < DON'T USE 'myData_171202-0037/' >
pathData3 = pathData2+'IMG/' # '../data/IMG' # <- to be updated with the AWS or Google path repo
pathData4 = pathData0+'logs/'
pathData5 = pathData4+'model/'
pathData6 = pathData4+'nn_logs/'
image_width  = 32 # 155  # 32
image_height = 32
pXl = 32
pxlRate = 0.01


def generator(center, left, right, steering, path=pathData3, batch_size=32, delta=0.25, image_width=image_width,image_height=image_height, pXl=pXl, pxlRate=pxlRate):
    num_lines = len(steering)
    while True: # Loop forever so the generator never terminates
        center, left, right, steering = shuffle(center, left, right, steering)
           
        for offset in range(0, num_lines, batch_size):
            # Import images and labels
            images, images_center, images_turn_left, images_turn_right, images_addon_right, images_addon_left = [],[],[],[],[],[]
            angles, angles_center, angles_turn_left, angles_turn_right, angles_addon_right, angles_addon_left = [],[],[],[],[],[]
            
            for batch_sample in range(offset,min(offset+batch_size,num_lines)):
                # Read steering angle and image
                angle        = float(steering[batch_sample])
                image_center = cv2.imread(pathData3 + center[batch_sample].split('\\')[-1])
                image_left   = cv2.imread(pathData3 +   left[batch_sample].split('\\')[-1])
                image_right  = cv2.imread(pathData3 +  right[batch_sample].split('\\')[-1])
                
                # Crop images > [65,320,3]  (to spare the RAM and speedup the computation)         
                image_center, image_left, image_right = image_center[70:-25, :, :], image_left[70:-25, :, :], image_right[70:-25, :, :]
                
                # Resize, http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
                image_center = cv2.resize(image_center, (image_width, image_height), cv2.INTER_AREA)
                image_left   = cv2.resize(image_left  , (image_width, image_height), cv2.INTER_AREA)
                image_right  = cv2.resize(image_right , (image_width, image_height), cv2.INTER_AREA)
                
                # Rebalance data: [0 : 'center', 1 : 'left', 2 : 'right', 3 : 'steering']
                if   angle > 0.15:
                    # Recovery: turn right > left camera
                    images_turn_right.append(image_left)
                    angles_turn_right.append(angle+delta)  
                    # Flip the image and adjust the steering angle: center vs (left, right)
                    images_addon_left.append(cv2.flip(images_turn_right[-1],1))
                    angles_addon_left.append(angles_turn_right[-1]*(-1.0))
                elif angle < -0.15:  
                    # Recovery: turn left > right camera
                    images_turn_left.append(image_right)
                    angles_turn_left.append(angle-delta)
                    # Flip the image and adjust the steering angle: center vs (left, right)
                    images_addon_right.append(cv2.flip(images_turn_left[-1],1))
                    angles_addon_right.append(angles_turn_left[-1]*(-1.0))
                else:
                    # Center > import images and output labels (steering angles)
                    images_center.append(image_center)
                    angles_center.append(angle)
                
            # Merge
            images_turn_right, angles_turn_right = images_turn_right + images_addon_right, angles_turn_right + angles_addon_right
            images_turn_left, angles_turn_left   = images_turn_left + images_addon_left, angles_turn_left + angles_addon_left
                     
            # Horizontal shift
            for i in range( int(len(images_center)-len(images_turn_right)) ):
                index = random.randint(0,len(images_center)-1)
                xPxl  = random.randint(-pXl, pXl) 
                
                M     = np.float32([[1,0,xPxl],[0,1,0]])
                image = cv2.warpAffine(images_center[index],M,(image_width,image_height))

                if   xPxl > 0:
                    images_turn_right.append(image)
                    angles_turn_right.append(angles_center[index]+pxlRate*xPxl)
                elif xPxl < 0:
                    images_turn_left.append(image)
                    angles_turn_left.append(angles_center[index]+pxlRate*xPxl)

            # print('')
            # print('images_center[]: {}, images_turn_right: {}, images_turn_left: {}'.format(len(images_center),len(images_turn_right), len(images_turn_left)))

            # Merge
            images = images_center + images_turn_left + images_turn_right 
            angles = angles_center + angles_turn_left + angles_turn_right
           
            # Randomly change brightness (to simulate day and night conditions) / hsv: hue, saturation, value
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                if np.random.rand() < 0.5:
                    image  = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    rate   = 1.0 + 0.4 * (np.random.rand() - 0.5)
                    image[:,:,2] =  image[:,:,2] * rate
                    image  = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                    # Add to the list
                    augmented_images.append(image)
                    augmented_angles.append(angle)
            #     # if np.random.rand() < 0.5:
            #     #     # Randomly shear the image
            #     #     delta = np.random.randint(-100,100)
            #     #     pts1  = np.float32([[0,image_height],[image_width, image_height],[image_width/2,image_height/2]])
            #     #     pts2  = np.float32([[0,image_height],[image_width, image_height],[image_width/2+delta,image_height/2]])   
            #     #     M     = cv2.getAffineTransform(pts1,pts2)
            #     #     image = cv2.warpAffine(image,M,(image_width, image_height),borderMode=1)
            #     #     angle+= delta/(image_height/2) * 360/(2*np.pi*25.0) / 6.0
            #     # Add to the list
            #     # augmented_images.append(image)
            #     # augmented_angles.append(angle)

            # # Convert images and labels into np.array
            X_train = np.array(augmented_images)  # X_train.shape = (143, 32, 155, 3)
            y_train = np.array(augmented_angles)
           

            # X_train, y_train = shuffle(X_train, y_train)

            yield shuffle(X_train, y_train)
