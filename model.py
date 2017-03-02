import os
import csv
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam

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



def get_images_url(url, folder, data_path):
    #parsing and getting the right url information:
    url_comp = url.replace('/',' ').replace('\\',' ').split(' ')
    url_comp[-1]
    new_url = data_path + folder + '/IMG/'+url_comp[-1]
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


def get_df_augmented(df):
    adjustment = 0.2
    df_augmented = pd.DataFrame([], columns = ['img_url', 'angle'])

    df['angle_left'] = df['steering'] + adjustment
    df['angle_right'] = df['steering'] - adjustment

    df_aux= pd.DataFrame({'img_url': df.center.tolist(), 'angle': df.steering.tolist()})
    df_augmented = pd.concat([df_augmented, df_aux], axis = 0)
    df_aux = pd.DataFrame({'img_url': df.left.tolist(), 'angle': df.angle_left.tolist()})
    df_augmented = pd.concat([df_augmented, df_aux], axis = 0)
    df_aux = pd.DataFrame({'img_url': df.right.tolist(), 'angle': df.angle_right.tolist()})
    df_augmented = pd.concat([df_augmented, df_aux], axis = 0)

    return(df_augmented)                           

def process_data(samples_df, training ):
    """
    The input is a dataframe in the log format from the simulator. 
    For each row in the log, we process each center, left and right image and we add these 3 and the corresponding flipped ones to a
    X, y arrays, that are the output of the function
    
    """
    if training:
        print('Training mode')
    num_obs = samples_df.shape[0]
    print('Num initial observations in the dataset: ', num_obs)
    image_url = data_log.img_url.tolist()
    angles = data_log.steering.tolist()

    # Preprocessing for each center image and angle in the data_log dataframe:
    X = []
    y = []

    for i in range(num_obs): # num_obs
        angle = angles[i]
        img = cv2.imread(image_url[i])
        img = preprocess_image(img)

        if abs(angle) < 0.1:
            # for small angles, we keep images with prob 50%
            if np.random.uniform() > 0.5:
                X.append(img)
                y.append(angle)
        else: 
            #if abs(angle)>0.15, we append the center image
            X.append(img)
            y.append(angle)


        # we just add flipped images and left and right images if angle > 0.33
        if abs(angle) > 0.33:
            #Adding center image flipped:
            img_flipped, angle_flipped = augmentation_flipping(img, angle)
            X.append(img_flipped)
            y.append(angle_flipped)        

 
    X = np.array(X)
    y = np.array(y)
    print('Len for processed datasets: ',len(X),len(y), X[0].shape)
    return(X,y)

def generator(X_samples,y_samples, batch_size):
    num_samples = len(X_samples)
    while 1: # Loop forever so the generator never terminates
        X_samples,y_samples = shuffle(X_samples,y_samples)
        for offset in range(0, num_samples, batch_size):
            X_batch_samples = X_samples[offset:offset+batch_size]
            y_batch_samples = y_samples[offset:offset+batch_size]

            yield sklearn.utils.shuffle(X_batch_samples, y_batch_samples)


## Local variables
plots_path = '/home/carnd/CarND-Behavioral-Cloning-P3/'
data_path = '/home/carnd/data/'


## UPLOADING DATA
##-----------------
print('...Data uploading...')
data_log = pd.DataFrame([], columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

for folder in ['data','slow','slow_part','extra_track1']: # 'recoverings','recov2', 'recov3', 'recov4' #'juanma',,'slow_part' #,'slow', 'juanma'
    filename = data_path + folder + "/driving_log.csv"
    print(filename)
    #read log data for the corresponding set of images:
    df = pd.read_csv(filename, sep = ",")
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    
    #Process urls for images, ensuring the right structure:
    df['center'] = df.apply(lambda x: get_images_url(x['center'], folder,data_path), axis = 1)
    df['left'] = df.apply(lambda x: get_images_url(x['left'], folder, data_path), axis = 1)
    df['right'] = df.apply(lambda x: get_images_url(x['right'], folder, data_path), axis = 1)
    
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

fig.savefig('steer_histogram_original_set.png')

## DATA AUGMENTATION WITH CENTER; LEFT AND RIGHT IMAGES
## ------------------
print('...Data augmentation for training set...')
data_augmented_df = get_df_augmented(data_log)
print('Shape for total augmented data set: ', data_augmented_df.shape)


## DATA SPLIT
##------------
# When X and y is big, sklearn.utils.shuffle runs out of RAM memory, using another approach:
# Splitting data into training and validation set from the data_log dataframe:
# Splitting data into training and validation set:
shuffle(data_augmented_df)
train_samples, validation_samples = train_test_split(data_augmented_df, test_size=0.2)

print('Len of train_samples: ', len(train_samples))
print('Len of validation_samples: ', len(validation_samples))


## DATA PREPROCESSING AND ADDING FLIPPING IMAGES
##-------------------
# For each training and validation set, we will obtain nd arrays with augmented data: center, left and right, and 
print('...Process training set...')
X_train,y_train = process_data(train_samples, training=True)
print('...Process validating set...')
X_val, y_val = process_data(validation_samples, training=False)

print('Training set processed: ',X_train.shape,y_train.shape)
print('validating set processed: ',X_val.shape, y_val.shape)



### Define the model
model = Sequential()

# Normalize data: Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x/127.5) - 1., input_shape=(66,200,3)))

#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Add three convolutional layers with a 2×2 stride and a 5×5 kernel, valid padding and filters: 24,36,48
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid',W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid',W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid',W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))


# Add 2 non-strided convolution with a 3×3 kernel size, valid padding and filters: 64,64
model.add(Convolution2D(64, 3, 3, border_mode='valid',W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid',W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))


# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers leading to an output control value which is the inverse turning radius
model.add(Dense(100,W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))
model.add(Dense(50,W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))
model.add(Dense(10,W_regularizer=l2(0.001)))
model.add(ELU()) # model.add(Activation('relu'))

# Add a fully connected output layer
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam') #Adam(lr=0.0001)



## Train the model:
print('...Training the network...')

EPOCHS = 10
batch_size = 128

# compile and train the model using the generator function
train_generator = generator(X_train,y_train, batch_size)
validation_generator = generator(X_val,y_val, batch_size)


history = model.fit_generator(train_generator, 
                    samples_per_epoch= len(X_train), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(X_val), 
                    nb_epoch=EPOCHS,verbose = 1)
print(model.summary()) 

## print the keys contained in the history object
#print(history.history.keys())
print('EPOCHS: ', EPOCHS)
print(history.history)

# Save model: creates a HDF5 file 'my_model.h5'
model.save('model.h5')
print('...model.h5 saved...')

### plot the training and validation loss for each epoch
fig =plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss \n')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
fig.savefig('Model_mse.png')



