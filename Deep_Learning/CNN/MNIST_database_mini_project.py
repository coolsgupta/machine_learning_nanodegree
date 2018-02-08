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

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

#summarise the model
model.summary()

