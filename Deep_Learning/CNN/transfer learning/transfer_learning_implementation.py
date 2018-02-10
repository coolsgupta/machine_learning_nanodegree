'''
Load and Preprocess Sample Images
Before supplying an image to a pre-trained network in Keras, there are some required preprocessing steps. You will learn
more about this in the project; for now, we have implemented this functionality for you in the first code cell of the
notebook. We have imported a very small dataset of 8 images and stored the preprocessed image input as img_input. Note
that the dimensionality of this array is (8, 224, 224, 3). In this case, each of the 8 images is a 3D tensor, with shape
(224, 224, 3).
'''


from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import glob

img_paths = glob.glob("images/*.jpg")

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

# calculate the image input. you will learn more about how this works the project!
img_input = preprocess_input(paths_to_tensor(img_paths))

print(img_input.shape)

#Recall how we import the VGG-16 network (including the final classification layer) that has been pre-trained on ImageNet.
from keras.applications.vgg16 import VGG16
model = VGG16()
model.summary()

model.predict(img_input).shape

'''
Import the VGG-16 Model, with the Final Fully-Connected Layers Removed
When performing transfer learning, we need to remove the final layers of the network, as they are too specific to the 
ImageNet database. This is accomplished in the code cell below.
'''

model = VGG16(include_top=False)
model.summary()

'''
Extract Output of Final Max Pooling Layer
Now, the network stored in model is a truncated version of the VGG-16 network, where the final three fully-connected 
layers have been removed. In this case, model.predict returns a 3D array (with dimensions  7×7×5127×7×512 ) corresponding 
to the final max pooling layer of VGG-16. The dimensionality of the obtained output from passing img_input through the 
model is (8, 7, 7, 512). The first value of 8 merely denotes that 8 images were passed through the network.
'''

print(model.predict(img_input).shape)

#This is exactly how we calculate the bottleneck features for your project!