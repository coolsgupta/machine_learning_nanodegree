import keras
from keras.datasets import cifar10

# load the pre-shuffled train and test data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# rescale [0,255] --> [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

from keras.utils import np_utils
import numpy as np
#one-hot encoding
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

# break training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# print shape of training set
print('x_train shape:', x_train.shape)

# print number of training, validation, and test images
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')

#build the mlp
from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(1000, activation='relu', ))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#compile the model
from keras.callbacks import ModelCheckpoint
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='MLP.weights.best.hdf5', verbose=1, save_best_only=True)

history = model.fit(x_train,y_train,batch_size=32, epochs=20, validation_data=(x_valid,y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)

# load the weights that yielded the best validation accuracy
model.load_weights('MLP.weights.best.hdf5')

# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])