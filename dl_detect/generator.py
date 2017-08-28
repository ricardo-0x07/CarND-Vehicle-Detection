import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split

samples = []
# Open and read csv file
with open('./data/set3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def resize_function(image):
    """Crop and Resize image.
    Args: image: image data
    Return: cropped and resized image 
    """
    image = image[60:140]
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    return  image

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    file_name = source_path.split('/')[-1]
                    current_path = './data/set3/IMG/' + file_name
                    image = cv2.imread(current_path)
                    if image is not None:
                        image = resize_function(image)
                        images.append(image)
                    else:
                        print('current path', current_path)

                center_angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.23 # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                angles.append(center_angle)
                angles.append(right_angle)
                angles.append(steering_right)
                augmented_images = []
                augmented_angles = []
                for image, angle in zip(images, angles):
                    augmented_images.append(image)
                    augmented_angles.append(angle)
                    # Flip images
                    augmented_images.append(cv2.flip(image, 1))
                    # Flip steering measurement
                    augmented_angles.append(angle*-1.0)


            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
