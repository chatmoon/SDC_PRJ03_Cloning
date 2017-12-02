import cv2, os
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle

pathData0 = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
pathData1 = pathData0+'data/'
pathData2 = pathData1+'myDebug/' # 'myDebug/' 'sample/'  'myData_171126-1643/'
pathData3 = pathData2+'IMG/' # '../data/IMG' # <- to be updated with the AWS or Google path repo
pathData4 = pathData0+'log/'
pathData5 = pathData4+'model/'

# # In[ X ]: Pre-process the Data Set
# def random_brightness(image):
#     """
#     Randomly change brightness (to simulate day and night conditions)
#     """
#     hsv  = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # hsv: hue, saturation, value
#     rate = 1.0 + 0.4 * (np.random.rand() - 0.5)
#     hsv[:,:,2] =  hsv[:,:,2] * rate
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# def random_flip(image, angle):
#     """
#     Randomly flipt the image and adjust the steering angle
#     """
#     if np.random.rand() < 0.5:
#         image = cv2.flip(image, 1)
#         angle = angle*(-1.0)
#     return image, angle

# In[ X ]: Pre-process the Data Set
def generator(lines, batch_size=32, delta=0.2):
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

            # Convert images and labels into np.array
            X_train = np.array(images)
            y_train = np.array(angles)

            # In[ X ]: DATA AUGMENTATION
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images,angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                # Randomly flipt the image and adjust the steering angle    
                if np.random.rand() <= 0.5:
                    image = cv2.flip(image, 1)
                    angle = angle*(-1.0)
                # Randomly change brightness (to simulate day and night conditions)
                    hsv  = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # hsv: hue, saturation, value
                    rate = 1.0 + 0.4 * (np.random.rand() - 0.5)
                    hsv[:,:,2] =  hsv[:,:,2] * rate
                    augmented_images.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
                    augmented_angles.append(angle)

            # Convert images and labels into np.array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)