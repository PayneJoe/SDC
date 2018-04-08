import os
import csv
import cv2
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.optimizers import Nadam

# LeNet architecture
def LeNet(learning_rate, keep_prob_1, keep_prob_2):
    # variables for le-net
    filter_1_num = 6
    filter_2_num = 16
    fc_1_num = 120
    fc_2_num = 84
    # model
    model = Sequential()
    # cropping
    model.add(Cropping2D(cropping= ((50, 20), (0, 0)), input_shape= (160, 320, 3)))
    # normaliation
    model.add(Lambda(lambda x: x/127.5 - 1.))
    # layer 1
    model.add(Conv2D(filter_1_num, kernel_size= (5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(1 - keep_prob_1))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    # layer 2
    model.add(Conv2D(filter_2_num, kernel_size= (5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(1 - keep_prob_2))
    model.add(MaxPooling2D(pool_size= (2, 2)))
    # layer 3
    model.add(Flatten())
    # layer 4
    model.add(Dense(fc_1_num))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # layer 5
    model.add(Dense(fc_2_num))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # layer 7
    model.add(Dense(1))
    model.compile(loss='mse', optimizer= Nadam(lr= learning_rate))
    
    model.summary()
    
    return model

# model architecture published by NVIDIA
def NVIDIA_CNN(learning_rate):
    # variables for le-net
    filter_1_num = 24
    filter_2_num = 36
    filter_3_num = 48
    filter_4_num = 64
    filter_5_num = 64
    fc_1_num = 100
    fc_2_num = 50
    fc_3_num = 10
    # model
    model = Sequential()
    # cropping, drop off 70 from the top and 25 from the bottom
    model.add(Cropping2D(cropping= ((70, 25), (0, 0)), input_shape= (160, 320, 3)))
    # normaliation
    model.add(Lambda(lambda x: x/127.5 - 1.))
    # convoluatinal layers
    model.add(Conv2D(filter_1_num, kernel_size= (5, 5), strides= (2, 2), activation= 'relu', name= 'CONV1'))
    model.add(Conv2D(filter_2_num, kernel_size= (5, 5), strides= (2, 2), activation= 'relu', name= 'CONV2'))
    model.add(Conv2D(filter_3_num, kernel_size= (5, 5), strides= (2, 2), activation= 'relu', name= 'CONV3'))
    model.add(Conv2D(filter_4_num, kernel_size= (3, 3), strides= (1, 1), activation= 'relu', name= 'CONV4'))
    model.add(Conv2D(filter_5_num, kernel_size= (3, 3), strides= (1, 1), activation= 'relu', name= 'CONV5'))
    # flatten the outputs of convolutional layer
    model.add(Flatten())
    # full connected layers
    model.add(Dense(fc_1_num))
    model.add(Dense(fc_2_num))
    model.add(Dense(fc_3_num))
    # output layer
    model.add(Dense(1))
    model.compile(loss='mse', optimizer= Nadam(lr= learning_rate))
    
    model.summary()
    
    return model

# super-parameters
keep_prob_1 = 0.5
keep_prob_2 = 0.8
batch_size = 32
epochs = 3
learning_rate= 0.0002
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples= samples[1:]
        
print('sample size %s' % len(samples))
# split original data set into train/valid with factor 0.2
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# using generator avoiding loading the entire data set at once
def generator(samples, mode= 'train', batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    # convert BGR supported by imread into RGB
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(image)
                    # using the three(left/center/right) image/angles to augment training data set
                    if(i == 0):
                        angles.append(center_angle)
                    elif(i == 1):
                        angles.append(center_angle + 0.2) # adjust the angel of left, factor 0.2 works well enough
                    else:
                        angles.append(center_angle - 0.2) # adjust the angel of right, factor 0.2 works well enough
#                 if(mode == 'train'):
#                     images.append(cv2.flip(center_image, 1))
#                     angles.append(center_angle * -1.0)

            X = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(X, y)

# compile and train the model using the generator function
train_generator = generator(train_samples, 'train', batch_size= batch_size)
validation_generator = generator(validation_samples, 'valid', batch_size= batch_size)

model = NVIDIA_CNN(learning_rate)
# model = LeNet(learning_rate, keep_prob_1, keep_prob_2)
model.fit_generator(train_generator, 
                    samples_per_epoch= len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), 
                    nb_epoch= epochs)
# model.summary()
model.save('model.h5')