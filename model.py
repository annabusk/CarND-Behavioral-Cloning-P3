import os
import csv
import cv2
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2, activity_l2

import matplotlib.image as mpimg
import scipy.ndimage
from scipy import stats
import cv2

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import getcwd



def get_images_url(url, folder):
    #parsing and getting the right url information:
    url_comp = url.replace('/',' ').replace('\\',' ').split(' ')
    url_comp[-1]
    new_url = folder + '/IMG/'+url_comp[-1]
    return(new_url)

def preprocess_image(img):
    '''
    Converts RGB images to YUV.
    Resizes original image shape: 160x320x3 to input shape for neural net network, 66x200x3
    '''
    new_img = img[50:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return(new_img)

def augmentation_flipping(img,angle):
    """
    Flippes images, multiplying by -1 the steering angle
    """
    image_flipped = np.fliplr(img)
    angle_flipped = -angle
    return(image_flipped,angle_flipped)

def data_visualization(X,y,y_pred):
    # Visualize data, steering andle and steering angle preducted, if informed, for exploratory purposes
    for i,image in enumerate(X):
        font = cv2.FONT_HERSHEY_PLAIN
        img = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        #print(img.shape)
        img = cv2.resize(img,None,fx=3, fy=3, interpolation = cv2.INTER_CUBIC)
        h,w = img.shape[0],img.shape[1]
        cv2.putText(img, str(i), org=(2,18), fontFace=font, fontScale=1., color=(255,0,0), thickness=2)
        cv2.putText(img, 'angle: ' + str(y[i]), org=(2,33), fontFace=font, fontScale=1., color=(255,0,0), thickness=2)
        cv2.line(img,(int(w/2),int(h)),(int(w/2+y[i]*w/4),int(h/2)),(0,255,0),thickness=4)      
        if len(y_pred) >0:
            cv2.line(img,(int(w/2),int(h)),(int(w/2+y_pred[i]*w/4),int(h/2)),(0,0,255),thickness=4)
        plt.figure()
        plt.imshow(img)  
    return

## Local variables
plots_path = '/home/carnd/CarND-Behavioral-Cloning-P3/'


## UPLOADING DATA
##-----------------
data_log = pd.DataFrame([], columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

for folder in ['data','slow']: # 'recoverings','recov2', 'recov3', 'recov4' #'juanma',,'slow_part'
    filename = folder + "/driving_log.csv"
    print(filename)
    #read log data for the corresponding set of images:
    df = pd.read_csv(filename, sep = ",")
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    
    #Process urls for images, ensuring the right structure:
    df['center'] = df.apply(lambda x: get_images_url(x['center'], folder), axis = 1)
    df['left'] = df.apply(lambda x: get_images_url(x['left'], folder), axis = 1)
    df['right'] = df.apply(lambda x: get_images_url(x['right'], folder), axis = 1)
    
    # Concatenate data frames:
    data_log = pd.concat((data_log, df), axis=0)


print('Data uploaded shape: ',data_log.shape)
print(data_log.head())


## DATA SET EXPLORATORY
##-------------------------
print('summary data for steering angle: ')
print(data_log.steering.describe())
# Plotting data histogram:

fig =plt.figure()
n, bins, patches = plt.hist(data_log['steering'], 20, align='left',   alpha=0.75)
plt.axvline(int(data_log['steering'].mean()), color='b', linestyle='dashed', linewidth=2)
plt.axvline(0, color='black', linestyle='dashed', linewidth=2)
plt.title('Histogram for steering angle data \n ')
plt.grid(True)
plt.show()

fig.savefig(plots_path + 'steer_histogram_original_set.png')

## DATA AUGMENTATION
##-------------------
num_obs = data_log.shape[0]
print('Num initial observations in the dataset: ', num_obs)
image_url = data_log.center.tolist()
image_left_url = data_log.left.tolist()
image_right_url = data_log.right.tolist()
angles = data_log.steering.tolist()


X = []
y = []
adjustment = 0.15
for i in range(num_obs): # num_obs
    
    #Adding center image and steering_angle:
    img = cv2.imread(image_url[i])
    angle = angles[i]
    img = preprocess_image(img)
    X.append(img)
    y.append(angle)
    #Adding center image flipped:
    img_flipped, angle_flipped = augmentation_flipping(img, angle)
    X.append(img_flipped)
    y.append(angle_flipped)

    #Adding left image:
    img = cv2.imread(image_left_url[i])
    angle = angles[i] + adjustment
    img = preprocess_image(img)
    X.append(img)
    y.append(angle)
    
    #Adding left image flipped:
    img_flipped, angle_flipped = augmentation_flipping(img, angle)
    X.append(img_flipped)
    y.append(angle_flipped)
    
    
    #Adding right image:
    img = cv2.imread(image_right_url[i])
    angle = angles[i] - adjustment
    img = preprocess_image(img)
    X.append(img)
    y.append(angle)
    
    #Adding right image flipped:
    img_flipped, angle_flipped = augmentation_flipping(img, angle)
    X.append(img_flipped)
    y.append(angle_flipped)   


X = np.array(X)
y = np.array(y)
print('Shapes for augmented datasets: ',len(X),len(y), X[0].shape)


