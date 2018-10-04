# **Behavioral Cloning** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project write up**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/center_2016_12_01_13_32_55_481.jpg "Center Lane Driving"
[image2]: ./examples/center_left_right.png "Left, Center and Right Camera Images"
[image3]: ./examples/crop/crop_3.png "Normal vs Cropped example"
[image4]: ./examples/flip/flip_3.png "Normal vs Flip example"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* final_model_1.h5 containing a trained convolution neural network 
* final_writeup_report_1.md summarizing the results
* README.md with overview of the project
* run_1.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the [NVIDIA End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model.
The model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 124-131) 
The model includes RELU layers to introduce nonlinearity (code line 124-131), and the data is normalized in the model using a Keras lambda layer (code line 119). 

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers. The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 57-78). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, using the left and right camera pictures with a steering correction factor of 0.2 and augmented the data set by flipping the images if the steering angle was greater than 0.15. I did not use data for 
recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use behavioral learning by using an existing network architecture and applying it to this problem. In the first step I used an example model from the class with a few convolutional and dense layers. This model had a lot of issues and the car seemed to have no control and it did not seem to me that any amount of tuning will get this to work. Therefore I decided to got with the convolution neural network model similar to the NDIIA model. I thought this model might be appropriate because a similar problem was sovled using the model on a real car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I modified the model so that the data is augmented by flipping the images with a high steer angle and using the left and right cameras.

My initial model had an issue in the generator which caused a significant slowdown due to an error in the yield statement. The final step was to run the simulator to see how well the car was driving around track one. The NVIDIA model was great and because of the data augmentation techniques, the training (0.0015) and validation loss (0.0174) were also low. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is the NVIDIA model architecture used in the project
Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3))

Convolution2D(24,5,5, subsample=(2,2), activation="relu"))

Convolution2D(36,5,5, subsample=(2,2), activation="relu")) 
Convolution2D(48,5,5, subsample=(2,2), activation="relu")) 
Convolution2D(64,3,3, activation="relu")) 
Convolution2D(64,3,3, activation="relu"))
Flatten()
Dense(100)
Dense(50)
Dense(10)
Dense(1)

| Layer					|Description										| 
|:---------------------:|:-------------------------------------------------:| 
| Input					| 32x32x1 RGB image   								| 
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 28x28x6		| #(32-5+1)/1 = 28
| RELU					|													|
| Max pooling			| 2x2 stride,  , VALID padding outputs 14x14x6		|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 10x10x16		| #(14-5+1)/1 = 10
| RELU					|													|
| Max pooling			| 2x2 stride,  , VALID padding outputs 5x5x16		|
| Flatten				| Output = 400										|
| Fully connected		| Input = 400. Output = 120							|
| RELU					|													|
| Fully connected		| Input = 120. Output = 84							|
| RELU					|													|
| Dropout				| keep_prob = 0.5									|
| Fully connected		| Input = 84. Output = 43							|
|						|													|

Here is the model summary report:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

The architecture can be visualized from the following link:
[NVIDIA End-to-End Deep Learning Model for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 

#### 3. Creation of the Training Set & Training Process

Since the recording using the keyboard was creating high steering angles and reduced the input data quality, I decided to use the data set from Udacity.
Here is an example image of center lane driving:

![Center Lane Driving][image1]

As suggested in the class, instead of using recovery data, I decided to take advantage of the left and right camera data by adding a steering correction.
The correction was (+0.2) for the left camera images and (-0.2) for the right camera images. This is a great alternative to recording data from left and right sides. These images show how the left, center and right camera images look like:

![Left, Center and Right Camera Images][image2]

After this I used the keras cropping function to crop the images by removing the sky and trees from the top and the hood from the bottom (model.py line 121). This improves the training by removing distracting features from the image. Here is an example image that illustrates the final result of cropping.

![Normal vs Cropped example][image3]

The car was mostly driving using left turns in the simulated path. To augment the data set, I flipped images and angles so that it can increase the quality of the training and validation datasets. For example, here is an image that has then been flipped:

![Normal vs Flip example][image4]


The data set had 8036 center images. After adding the left and right images, the size of data points was 24108. Then flipping the images added more data to this. Right in the beginning, I split the data into training(80%) and validation(20%). I randomly shuffled the data before starting the generator and also in the yield statement. 
Given the size of the data set, using the generator greatly improved the performance. I passed the image paths in and returned the loaded image array.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the results. I used an adam optimizer so that manually training the learning rate wasn't necessary.
