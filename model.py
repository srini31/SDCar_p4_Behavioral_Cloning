'''
data is corrected for multiple cameras and augmented

data is
- collected for three cameras
- steering values are corrected for left and right cameras
- augmented by flipping images if abs(angle) > 0.15
'''

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

#from utility import utility
#import pickle
#utilityClass = utility()

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if (line[3] == 'steering'):  #ignore the first line
            continue
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('sample size, train, valid: ', len(train_samples), '/', len(validation_samples))

#IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        #for offset in range(0, num_samples, batch_size):
            #batch_samples = samples[offset:offset+batch_size]

        images = []
        angles = []
        #images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        #angles = np.empty(batch_size)
        correction = 0.2  # this is a parameter to tune
        count = 0
        for batch_sample in samples:  #for batch_sample in batch_samples:
            steering_angle = float(batch_sample[3])
            for idx in range(3):
                name = 'data/IMG/'+batch_sample[idx].split('/')[-1]
                car_image = cv2.imread(name)
                images.append(car_image)
                #--------------
                if(idx == 0): #center
                    final_angle = steering_angle
                # use existing measurement
                elif (idx == 1):  # left
                    final_angle = steering_angle + correction
                elif (idx == 2):  # right
                    final_angle = steering_angle - correction

                angles.append(final_angle)  # center
                count = count + 1
                #index order is center, left, right
                #---------------------------------------
                #augment the images
                # only flip images where steering angle is > abs(0.15)
                if (abs(final_angle) > 0.15):
                    images.append(cv2.flip(car_image, 1))
                    angles.append(final_angle * -1.0)
                    count = count + 1

                #print('len(images): ', len(images), '/',  'len(angles): ', len(angles))

            # reset images once it has more images than batch size
            # it cant be exact as we are also flipping and we cant keep track of count
            if (len(images) >= batch_size) :  #if len(images) == batch_size:  #if count % batch_size  == 0:
                X_train = np.array(images)
                y_train = np.array(angles)
                #print('Shape X_train: ', X_train[0].shape, ' y_train: ', y_train.shape)
                #print('count: ', count, ' Shape X_train: ', X_train[0].shape, 'Shape X_train: ', X_train.shape, ' y_train: ', y_train.shape)
                #count = 0
                yield sklearn.utils.shuffle(X_train, y_train)
                images = []
                angles = []

        #yield sklearn.utils.shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#print('len X_train: ',  len(images), ' y_train: ',len(measurements))
#print('image[0].shape: ', len(image[0]))
#X_train = np.reshape(len(X_train),160, 320, 3)
#print('Shape X_train: ',  X_train[0].shape, ' y_train: ',y_train.shape)


#start the keras network
ch, row, col = 3, 160, 320  #3, 80, 320  # Trimmed image format

#from keras import backend as K
#K.set_image_dim_ordering('th')
# https://stackoverflow.com/questions/41651628/negative-dimension-size-caused-by-subtracting-3-from-1-for-conv2d

#use the nvidia network @ https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
#cropping layer to remove top(sky, trees) and bottom (hood) of the image_flipped
model.add(Cropping2D(cropping=((70,25), (0,0))))  # input_shape=(3,160,320)
#70 rows pixels from the top of the image, 25 rows pixels from the bottom of the image
#0 columns of pixels from the left and right  of the image
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu")) #6,5,5
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu")) #6,5,5
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu")) #6,5,5
#dropout here or at any other conv layer
model.add(Convolution2D(64,3,3, activation="relu")) #6,5,5
model.add(Convolution2D(64,3,3, activation="relu")) #6,5,5
model.add(Flatten())
model.add(Dense(100))
#dropout can be added here if necessary
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples*4), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples*4), nb_epoch=3, verbose=1)

print('model summary: ', model.summary())
#the generator keeps repeating the data, samples_per_epoch will limit how many samples are actually used in an epoch
# after an epoch, the generator resets and starts from beginning
# train_samples*4 is used because if we add the left, right and flipped images to center ones, its *4

#If the above code throw exceptions, try
#model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
#  validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

model.save('model_final_v1.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('training_info')