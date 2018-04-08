# **Behavioral Cloning** 

## The goal and steps

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results
* video.mp4 generated on test data set in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

* I finally used the CNN architecture published by NVIDIA. The complete architecture of it is shown as below:


| Layer             |     Description	            | 
|:-----------------:|:-----------------------------:| 
| Input             | 160x320x3 RGB image   		|
| Cropping Layer    | kernel ((70, 25), (0, 0))     |
| Lambda Layer      | Normalization,range[-1,1]     |
| Conv Layer1(24)   | kernel=(5,5),strides=(2,2),RELU|
| Conv Layer2(36)   | kernel=(5,5),strides=(2,2),RELU|
| Conv Layer3(48)   | kernel=(5,5),strides=(2,2),RELU|
| Conv Layer4(64)   | kernel=(3,3),strides=(1,1),RELU|
| Conv Layer5(64)   | kernel=(3,3),strides=(1,1),RELU|
| Flatten Layer     |                               |
| FullConn Layer(100)| full connected layer         |
| FullConn Layer(50) | full connected layer         |
| FullConn Layer(10) | full connected layer         |
| FullConn Layer(1)  | Output Layer                 |

It is characterized by 5 convolutional layers with different filter number and same kernel, and 3 fully connected layers with different output size, and the last output fully connected layer.

* I also tried the LeNet architecture used in Traffic Sign Classification project, though not good enough on this project for me by now:

|Layer                | Description                  |
|:-------------------:|:----------------------------:|
|Input                | 160\*320\*3 RGB              |
|Cropping Layer       | kernel ((70, 25), (0,0))     |
|Lambda Layer         | Normalization, range[-1,1]   |
|Conv Layer1(6)       | kernel=(5,5),strides=(1,1),RELU|
|Dropout Layer        | keep_proba=0.5               |
|MaxPooling Layer     | kernal=(2,2)                 |
|Conv Layer2(16)       | kernel=(5,5),strides=(1,1),RELU|
|Dropout Layer        | keep_proba=0.8               |
|MaxPooling Layer     | kernal=(2,2)                 |
|Flatten Layer        |                              |
|FullConn Layer(120)  | full connected layer, RELU   |
|FullConn Layer(84)   | full connected layer, RELU   |
|FullConn Layer(1)    | output layer                 |

It is characterized by 2 convolutional layers with different filter number and same kernel, and 2 fully connected layers with different output size, and the last output fully connected layer.

#### 2. Attempts to reduce overfitting in the model

* I have not used dropout or pooling layers on NVIDIA_CNN architecture to reduce overfitting by now, since the losses on train and valid data set look good enough during the three epochs(both drops):

```
Epoch 1/3
6428/6428 [===============] - 481s - loss: 0.0046 - val_loss: 0.0224
6428/6428 [===============] - 476s - loss: 0.0011 - val_loss: 0.0199
6428/6428 [===============] - 478s - loss: 0.0010 - val_loss: 0.0171
```
More importantly it works pretty well in autonomous mode on test data set.

* I used dropout, pooling layers to reduce overfitting on LeNet model architecture.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

Parameters of CNN published by NVIDIA: 

```
batch_size = 32
epoch = 3
learning_rate = 0.0002
```

* The model used an nadam optimizer, I chose 0.0002 after few times trying.
* 3 is good enough for epoch
* 32 is a good option for speed, bigger seems more slower, generator is a time consuming operation 

#### 4. Appropriate training data

#### Provided data set

I failed to generate high quality data set as my mouse seemed not able to take control of the steering.

So at first I had to use the data set generated by keyboard. In the result, my car was not able to turn left/right at all after few hours adjustment(preprocessing, augmentation) on my data sets and refinement on my model architecture.

Finally, I used the data set provided on project main page instead after noticing the tips about that on slack. I seems like my data set created with keyboard is **Garbage**, so the output is **Garbage** too.

#### Augmented data set
I used the three(left/center/right) images as my entire data sets, chose 0.2 as the angle adustment factor for left/right, and it seemed preety good enough.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

* At the very first, I tried to use LeNet as my primary model strategy with few loops of custom data sets whose steering data was generated by keyboard since my mouse was not able to control the steering. The result of it is pretty bad, the car could not even turn left/right at corners.

* Then, I also used LeNet as my primary model strategy with the data set provided on project main page, and tried dropout/pooling layers to reduce overfitting. The car worked even better, unfortunately failed to turn left at the end of bridge even after I joined the left/right images.

* At last, I tried to use the NVIDIA_CNN. It also failed to turn left at the end of bridge before adding the left/right images. Fortunately it worked pretty good after join the left/right images.

#### 2. Final Model Architecture

My final architecture is the CNN published by NVIDIA whose detail has already been presented above. 

#### 3. Creation of the Training Set & Training Process

I failed to create high quality custom data set. The details of training process has already been presented above. Mainly four steps:
* try custom data set, failed to turn left/right due to its low quality of it
* try provided data set, regularized LeNet architecture, event better, but failed to turn left at the end of bridge
* try adding left/right images as augmentation, also not work well
* try CNN published by NVIDIA, it worked good enough at last

## Further exploration
* modification on CNN architecture published by NVIDIA
* more augmentation strategies, such flipping/brightness
* more advanced CNN architecture

## Lesson learned from this project
* quality of data always being the most primary factor we should take into consideration in this project, as always saying "garbage in, garbage out"
* different angle of images is a good option for data augmentation