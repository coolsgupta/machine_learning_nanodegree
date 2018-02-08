#import data
from keras.datasets import mnist

(x_train, y_train),(x_test,y_test) = mnist.load_data()

print("Training set size : %d" %len(x_train))
print("Test set size : %d" %len(x_test))

#reshape data
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

from keras.utils import np_utils
# print first ten (integer-valued) training labels
print('Integer-valued labels:')
print(y_train[:10])

# one-hot encode the labels
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

# print first ten (one-hot) training labels
print('One-hot labels:')
print(y_train[:10])

#we create a CNN with 784 nodes in input layer because we have 784 entries in our input vector since we converted the 28x28 matrix to a single vector

#creating the model
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten

#define the model
model = Sequential()
#add layers
model.add(Flatten(input_shape=x_train.shape[1:])) #the flatten layer takes the image matrix inputs and converts it into a vector
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

#summarise the model
model.summary()

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#evaluate accuracy before training
score = model.evaluate(x_test,y_test,verbose=0)
accuracy = score[1]*100
print('\n\nTest accuracy without training :  %d %%' %accuracy)

#add checkpointer to save the model weights at specified location
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

#train the model
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)

# load the weights that yielded the best validation accuracy
model.load_weights('mnist.model.best.hdf5')

# evaluate test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('\n\nTest accuracy: %.4f%%' % accuracy)