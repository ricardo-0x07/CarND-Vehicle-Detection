import csv
import cv2
import numpy as np
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dropout, Activation, Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split

samples = []
# Open and read csv file
with open('./data/set3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

import os
import glob
dirs = os.listdir("../data/vehicles/")
cars = []
print(dirs)
for image_type in dirs:
    cars.extend(glob.glob('../data/vehicles/'+ image_type+'/*'))
    
print('Number of Vehicles Images found', len(cars))


dirs = os.listdir("../data/non-vehicles/")
notcars = []
print(dirs)
for image_type in dirs:
    notcars.extend(glob.glob('../data/non-vehicles/'+ image_type+'/*'))
    
print('Number of Non-Vehicles Images found', len(notcars))
y = np.hstack((np.ones(len(cars)),np.zeros(len(notcars))))
X = np.vstack((cars, notcars)).astype(np.float)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

def resize_function(image):
    """Crop and Resize image.
    Args: image: image data
    Return: cropped and resized image 
    """
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    return  image

def generator(X, y, batch_size=32):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]

            images = []
            angles = []
            for source_path in batch_X:
                image = cv2.imread(source_path)
                images.append(image)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(y)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=1024)
validation_generator = generator(X_test, y_test, batch_size=1024)

model = Sequential()
# Normalize image data
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(64,64,3)))
# Add convolution layers
model.add(Conv2D(6, (5, 5), strides=1, padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (5, 5), strides=1, padding='same'))  
model.add(Activation('elu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (3, 3), strides=1, padding='same'))  
model.add(Activation('elu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (3, 3), padding='same'))  
model.add(Activation('elu'))
model.add(MaxPooling2D())
# Add flatten layer
model.add(Flatten())
# Add dense layers
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.20))

model.add(Dense(50, activation='elu'))
model.add(Dropout(0.20))

model.add(Dense(10, activation='elu'))
model.add(Dropout(0.5))

model.add(Dense(1), activation='sigmoid')
# Print model summary
model.summary()
# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Run model
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10)
# Save model
model.save('model.h5')
