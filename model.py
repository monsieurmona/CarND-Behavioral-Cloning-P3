import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# network includes
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
#from keras.preprocessing.image import ImageDataGenerator

recording_path = '/home/mona/src/udacity/simulator/linux_sim/recording'
#recording_path = '/home/mona/src/udacity/CarND-Behavioral-Cloning-P3/sample-training-data/data'


def plot_histogram(data, n_bins):
    histogram, bins = np.histogram(data, bins=n_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, histogram, align='center', width=width)


def add_to_historgram(value, max, histogram, bounds):
    for i in range(len(bounds) - 1):
        if (value >= bounds[i] and value <= bounds[i+1]):
            if histogram[i] > max:
                return True
            else:
                histogram[i] += 1
                return False
    return False


# convert image to yuv, with tensorflow
def rgb_image_to_yuv_conversion(x):
    import tensorflow as tf
    return tf.image.rgb_to_yuv(x)

def per_image_standardization(x):
    import tensorflow as tf
    return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)


# load csv
column_names = ['center', 'left', 'right',
                'steering', 'throttle', 'brake', 'speed']
#lines = pd.read_csv(recording_path + '/driving_log.csv', names=column_names, nrows=200)
lines = pd.read_csv(recording_path + '/driving_log.csv', names=column_names)

# shuffle
n_histogram_bins = 50
lines = lines.sample(frac=1).reset_index(drop=True)
histogram, bin_bounds = np.histogram(lines['steering'], bins=n_histogram_bins)
histogram.fill(0)

images = []
measurements = []

for idx, line in lines.iterrows():
    # load steering measurments
    steering_measurement = float(line[3])

    correction = 0.05

    for image_idx in range(3):
        source_path = line[image_idx]

        # no need to change path name as recording and training is done on the same machine
        filename = source_path.split('/')[-1]
        current_path = recording_path + '/IMG/' + filename
        # current_path = source_path

        steering = steering_measurement

        # if left image steer to the left
        if image_idx == 1:
            steering = steering_measurement + correction

        # if right image steer to the
        if image_idx == 2:
            steering = steering_measurement - correction

        if add_to_historgram(steering, 3500, histogram, bin_bounds) == True:
            continue

        measurements.append(steering)

        # load the image
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

        # add flipped
        images.append(np.fliplr(image))
        measurements.append(- steering)

X_train = np.array(images)
y_train = np.array(measurements)


# show distribution of steering angles
# histogram of label frequency

plot_histogram(y_train, n_histogram_bins)
#plot_histogram(lines['steering'], n_histogram_bins)
plt.show()
'''
'''

input_shape= X_train.shape[1:]
print("Training elements:" + str(len(X_train)))
print("Input shape:" + str(input_shape))
nb_epoch = 10
batch_size = 10

# regression network - not a classification network
#for i in range(3,8):
#    print("-----------------------------------")
#    print("Dense:" + str(pow(2,i)))
#    print("-----------------------------------")
model = Sequential()

# cropping
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=input_shape))

# normalize data and mean centering
#model.add(Lambda(lambda  x: x / 255.0 - 0.5))
model.add(Lambda(per_image_standardization))

# image conversion to yuv
model.add(Lambda(rgb_image_to_yuv_conversion))

# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
# Convolutional layer with 2x3 stride and 5x5 kernel
model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation="elu"))
model.add(Conv2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation="elu"))
model.add(Conv2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation="elu"))

# Convolutional layer without stride and 3x3 kernel
model.add(Conv2D(64, 3, 3, border_mode='valid', activation="elu"))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation="elu"))


# Fully connected layer
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# mse = mean squared error
model.compile(loss='mse', optimizer='adam')

'''
#augument images
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

datagen.fit(X_train)

#history_object = model.fit_generator(datagen.flow(X_train.astype('float32'), y_train, batch_size=batch_size), samples_per_epoch=len(X_train), nb_epoch=nb_epoch, validation_data=(X_valid, y_valid), verbose=1)
'''

history_object = model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, verbose=1)

model.save("nvidia_network_fit_generator.h5")

# show loss
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()