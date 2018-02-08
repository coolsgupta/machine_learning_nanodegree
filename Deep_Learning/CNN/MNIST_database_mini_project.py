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