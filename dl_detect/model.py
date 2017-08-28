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
import os
import glob
dirs = os.listdir("../data/vehicles/")
cars = []
print(dirs)
for image_type in dirs:
    cars.extend(glob.glob('../data/vehicles/'+ image_type+'/*.jpg'))
    
print('Number of Vehicles Images found', len(cars))


dirs = os.listdir("../data/non-vehicles/")
notcars = []
print(dirs)
for image_type in dirs:
    notcars.extend(glob.glob('../data/non-vehicles/'+ image_type+'/*.jpg'))
    
print('Number of Non-Vehicles Images found', len(notcars))
y = np.concatenate((np.ones(len(cars)),np.zeros(len(notcars))))
X = np.concatenate((cars, notcars))
print('X.shape',X.shape)
print('y.shape',y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

def resize_function(image):
    """Crop and Resize image.
    Args: image: image data
    Return: cropped and resized image 
    """
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    return  image

def generator(X, y, batch_size=1024):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
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
            y_train = np.array(batch_y)
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

model.add(Dense(1, activation='sigmoid'))
# Print model summary
model.summary()
# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Run model
model.fit_generator(train_generator, steps_per_epoch=len(X_train)/1024, validation_data=validation_generator,validation_steps=len(X_test)/1024, epochs=100)
# Save model
model.save('model.h5')
