#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:15:23 2018

@author: dhingratul
"""

# Imports
from __future__ import print_function
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from IPython.display import clear_output
from keras import backend as K
import getData
import numpy as np
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import plot_model

# input image dimensions
img_rows, img_cols = 100, 100
# Data
train_dir = 'data/train/'
x_train, y_train, _ = getData.load_data(train_dir, img_rows, img_cols)
print("Training data done")
test_dir = 'data/test/'
x_test, y_test, x_images = getData.load_data(test_dir, img_rows, img_cols)
print("Testing data done")

# Clean up data imbalance
x_train, x_test, y_train, y_test = getData.imbalance(x_train, x_test, y_train, y_test)
# Model Parameters
batch_size = 64
y_all = np.concatenate((y_train, y_test))
num_classes = len(np.unique(y_all))
epochs = 35                                                                                                                                                                                                                                                                                                                             

# Convert images into proper format that keras reads
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
# Data Normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Label Encoder
le = preprocessing.LabelEncoder()
le.fit(np.unique(y_all))
y_train = le.transform(y_train) 
y_test = le.transform(y_test) 

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Plot Loss in real time 
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig("figs/{}.png".format(epoch))
        plt.show();
        
plot_losses = PlotLosses()


#  Model Definition
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Model Summary
model.summary()
# Train Model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Train Model
train = True
datagen = False
if train is True:
    if datagen is True:
        datagen = ImageDataGenerator(
                featurewise_center=False,
                zca_whitening=False,
                shear_range=0.,
                rotation_range=0.,
                featurewise_std_normalization=False,
                width_shift_range=0,
                height_shift_range=0,
                horizontal_flip=True)
     
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                        validation_data=(x_test, y_test))
#                        ,callbacks=[plot_losses])
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
#                  ,callbacks=[plot_losses])
    model.save('model/model.h5')
    plot_model(model, to_file='img/model.png')
else:
    model = load_model('model/model.h5')


# +1/-1 metric on Test Set
print("+1/-1 metric on Test Set")
_, _, perf = getData.metric(model, x_test, y_test, y_test.shape[0], le)
print(perf)