# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:55:27 2017

@author: lenovo
"""

#Convolutional Neural Networks
#since we are using data which is already managed very well and we dont need to work out on splitting the training and testing sets.
#working on building the cnn will be a good idea and this will be our 1st part of our program.
#Part-1 building the cnn
#importing the required libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN as a sequence of layers
classifier = Sequential()
 
#Adding the Layers
#Convolution -> Max Pooling -> Flattening -> Full Connection
#Step-1 -> Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))
#Step-2 -> Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
#Step-3 -> Flattening
classifier.add(Flatten())
#step-4 -> Full Connection
#output nodes here can be decided on the basis of input nodes and output nodes as an average 
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#Compiling the Model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#fitting the Model
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'dataset/training_set', target_size=(64, 64), batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
