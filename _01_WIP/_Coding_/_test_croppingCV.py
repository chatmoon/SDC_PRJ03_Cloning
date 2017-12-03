import csv
import cv2, os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle

pathData0 = 'C:/Users/mo/home/_eSDC2_/_PRJ03_/_2_WIP/_171126-1433_BehavioralCloning/'
pathData1 = pathData0+'data/'
pathData2 = pathData1+'myDebug/' # 'myDebug/' 'sample/'  'myData_171126-1643/'
pathData3 = pathData2+'IMG/' # '../data/IMG' # <- to be updated with the AWS or Google path repo

lines = []
with open(pathData2+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Import images and labels
images = []
angles = []

# Update the localpath with the cloudpath
source_path = lines[0][0] # source_path = line[i] # local path 
filename = source_path.split('\\')[-1] # ('/')[-1] # file name
current_path = pathData3 + filename

# Import images
image = cv2.imread(current_path)

# Crop images
#cropping_output = backend.function([model.layers[0].input],[model.layers[0].output])
#cropped_image   = cropping_output([image[None,...]])[0]
cropped_image = image[70:-25, :, :]
resized_image = cv2.resize(cropped_image, (32, 155), cv2.INTER_AREA)

def compare_images(left_image, right_image):    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    # Image 1
    ax1.imshow(left_image)
    ax1.set_title('Shape '+ str(left_image.shape), fontsize=50)
    # Image 2
    ax2.imshow(np.uint8(right_image))
    ax2.set_title('Shape '+ str(right_image.shape), fontsize=50)
    plt.show()


#compare_images(image,cropped_image)
compare_images(cropped_image,resized_image)
