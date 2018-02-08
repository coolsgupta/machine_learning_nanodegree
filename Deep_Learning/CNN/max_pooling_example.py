from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
model.summary()

'''

Max Pooling Layers in Keras
To create a max pooling layer in Keras, you must first import the necessary module:

from keras.layers import MaxPooling2D
Then, you can create a convolutional layer by using the following format:

MaxPooling2D(pool_size, strides, padding)
Arguments
You must include the following argument:

pool_size - Number specifying the height and width of the pooling window.
There are some additional, optional arguments that you might like to tune:

strides - The vertical and horizontal stride. If you don't specify anything, strides will default to pool_size.
padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.
NOTE: It is possible to represent both pool_size and strides as either a number or a tuple.

You are also encouraged to read the official documentation.

Example
Say I'm constructing a CNN, and I'd like to reduce the dimensionality of a convolutional layer by following it with a
 max pooling layer. Say the convolutional layer has size (100, 100, 15), and I'd like the max pooling layer to have size 
 (50, 50, 15). I can do this by using a 2x2 window in my max pooling layer, with a stride of 2, which could be
  constructed in the following line of code:

    MaxPooling2D(pool_size=2, strides=2)
If you'd instead like to use a stride of 1, but still keep the size of the window at 2x2, then you'd use:

    MaxPooling2D(pool_size=2, strides=1)

'''