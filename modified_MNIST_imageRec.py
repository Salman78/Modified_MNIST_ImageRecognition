'''
import pickle
import numpy as np 
import pandas as pd 

train_images = pd.read_pickle('/Users/Tausal21/Desktop/comp_551/mini_project3/train_images.pkl')
train_labels = pd.read_csv('/Users/Tausal21/Desktop/comp_551/mini_project3/train_labels.csv')

train_images_numpy = np.array(train_images)
train_labels_numpy = np.array(train_labels)
'''
import numpy as np

numpy_arr = np.zeros((10))
print(numpy_arr.shape)
'''
for i in range(2):	
	numpy_arr[i].fill(8)

print(numpy_arr)

random_3by3_array = np.ones((3,5))
#print(random_3by3_array)

for m in range(2):
	for x in range(3):
		for y in range(5):
			numpy_arr[m].fill(random_3by3_array[x][y])
print(numpy_arr)
'''	







'''
#cell 1
train_images_numpy = np.stack((train_images_numpy,) * 1, axis=-3)

train_labels_dim = train_labels_numpy.shape
train_labels_only = np.zeros(train_labels_dim[0])
for i in range(train_labels_dim[0]):
    train_labels_only[i] = train_labels_numpy[i][1]
    

print("train images dimension: ",  train_images_numpy.shape)
print("train labels only dimension: ",  train_labels_only.shape)

print("train images: ", train_images_numpy)
print("train labels only: ", train_labels_only)

#cell2
X_train, X_validation, Y_train, Y_validation = train_test_split(train_images_numpy, train_labels_only, test_size=0.25, random_state=42)

print("X_train dimension: ",  X_train.shape)
print("X_train: ",  X_train)
print("Y_train dimension: ",  Y_train.shape)
print("Y_train: ",  Y_train)

print("X_validation dimension: ",  X_validation.shape)
print("X_validation: ",  X_validation)
print("Y_validation dimension: ",  Y_validation.shape)
print("Y_validation: ",  Y_validation)

X_train = keras.utils.normalize(X_train, axis=3)
X_validation = keras.utils.normalize(X_validation, axis=3)

print("X_train normalized: ",  X_train)
print("X_validation normalized: ",  X_validation)


#cell3
train_batches = ImageDataGenerator(data_format="channels_first").flow(X_train, Y_train, batch_size=3000, shuffle=True)
validation_batches = ImageDataGenerator(data_format="channels_first").flow(X_validation, Y_validation, batch_size=1000, shuffle=True)

model = Sequential([
    Conv2D(8, (3,3), activation='relu', data_format="channels_first", input_shape=(1,64,64)),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=10, validation_data=validation_batches, validation_steps=10, epochs=5, verbose=2)

#commented out block

scaler = MinMaxScaler(feature_range=(0,1))

train_images_dim = train_images_numpy.shape

for x in range(train_images_dim[0]):
    scaled_train_images = scaler.fit_transform(train_images_numpy[x])
for x in range(train_images_dim[0]):
    for m in range(train_images_dim[1]):
        for n in range(train_images_dim[2]):
            train_images_numpy[x].fill(scaled_train_images[m][n])
print(train_images_numpy)

'''


