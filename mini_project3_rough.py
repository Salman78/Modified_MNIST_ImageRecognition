import pickle
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, AvgPool2D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import ReduceLROnPlateau
from keras.utils import normalize
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
%matplotlib inline

train_images = pd.read_pickle('/Users/Tausal21/Desktop/comp_551/mini_project3/train_images.pkl')
train_labels = pd.read_csv('/Users/Tausal21/Desktop/comp_551/mini_project3/train_labels.csv')

train_labels_only = []
train_labels_only = train_labels['Category'].values.tolist()
train_labels_only = np.array(train_labels_only)
#train_labels_only = train_labels.drop(labels = ["Id"],axis = 1)
#train_labels_only = train_labels.drop(columns = ["Id"])
print(train_labels_only.shape)

train_images = train_images/255.0
print(train_images.shape)

train_images = np.array(train_images)
train_labels_only = to_categorical(train_labels_only, num_classes = 10)

print(train_labels.shape)

train_images = train_images.reshape(-1,64,64,1)
print(train_images.shape)


X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels_only, test_size = 0.25, random_state=2)
print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)



sns.set(style='white', context='notebook', palette='deep')
g = plt.imshow(X_train[0][:,:,0])



'''
train_batches = ImageDataGenerator(
    data_format="channels_last",
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True).flow(X_train, Y_train, batch_size=30, shuffle=True)
validation_batches = ImageDataGenerator(
    data_format="channels_last",
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True).flow(X_val, Y_val, batch_size=10, shuffle=True)
'''
'''
model = Sequential([
    Conv2D(8, (3,3), activation='relu', data_format="channels_last", input_shape=(64,64,1)),
    Flatten(),
    Dense(10, activation='softmax')
])
'''
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))
'''
'''
model.compile(Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

#model.fit_generator(train_batches, steps_per_epoch=10, validation_data=validation_batches, validation_steps=10, epochs=10, verbose=2)
model.fit(X_train, Y_train,
          batch_size=1000,
          epochs=25,
          verbose=2,
          validation_data=(X_val, Y_val))
'''

'''
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(Adam(lr=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5)
'''



model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(AvgPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 30
batch_size = 1000

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val),callbacks=[learning_rate_reduction])


