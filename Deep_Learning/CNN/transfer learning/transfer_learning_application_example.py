#Load Dataset
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target'], 133))
    return dog_files,dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load ordered list of dog names
dog_names = [item[25:1] for item in glob('dogImages/train/*/')]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % str(len(train_files) + len(valid_files) + len(test_files)))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

'''
Obtain the VGG-16 Bottleneck Features
Before running the code cell below, download the file linked here and place it in the bottleneck_features/ folder.
'''

bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_vgg16 = bottleneck_features['train']
valid_vgg16 = bottleneck_features['valid']
test_vgg16 = bottleneck_features['test']

#Define a Model Architecture (Model 1)

from keras.layers import Dense, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Flatten(input_shape=(7, 7, 512)))
model.add(Dense(133, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])
model.summary()

#Define another Model Architecture (Model 2)

from keras.layers import GlobalAveragePooling2D

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(7,7,512)))
model.add(Dense(133,activation='softmax'))
model.summary()

#Compile the Model (Model 2)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the Model (Model 2)
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='dogvgg16.weights.best.hdf5', verbose=1, save_best_only=True)

model.fit(train_vgg16, train_targets, epochs=20, validation_data=(valid_vgg16, valid_targets), verbose=1, callbacks=[checkpointer], shuffle=True)

# load the weights that yielded the best validation accuracy
model.load_weights('dogvgg16.weights.best.hdf5')

#Calculate Classification Accuracy on Test Set (Model 2)

# get index of predicted dog breed for each image in test set
vgg16_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0)))
                     for feature in test_vgg16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(vgg16_predictions)==np.argmax(test_targets, axis=1))/len(vgg16_predictions)
print('\nTest accuracy: %.4f%%' % test_accuracy)