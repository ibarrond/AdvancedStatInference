from utils import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
K.set_image_dim_ordering('th')

mnist_X_train, mnist_y_train = load_mnist(dataset='training')
mnist_X_test,  mnist_y_test  = load_mnist(dataset='testing')

from sklearn.preprocessing import OneHotEncoder

# Converts target labels into one-hot vectors.
# This is needed since our target labels are categorical (not continuous).
def preprocess_labels(labels):
    enc = OneHotEncoder()
    converted_labels = np.array(labels).reshape(-1, 1)
    oh_labels = enc.fit_transform(converted_labels).toarray()
    
    return oh_labels


 
class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        # initialize the model
        model = Sequential()
        # first set of CONV => RELU => POOL
        model.add(Conv2D(6, (5, 5), padding="same",
            input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL
        model.add(Conv2D(16, (5, 5), padding="valid"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        # second set of FC => RELU layers
        model.add(Dense(84))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model


training_features = np.array([feature / 255.0 for feature in mnist_X_train])
training_features = training_features[:, np.newaxis, :, :]
test_features = np.array([feature / 255.0 for feature in mnist_X_test])
test_features = test_features[:, np.newaxis, :, :]
training_labels = preprocess_labels(mnist_y_train)
test_labels = preprocess_labels(mnist_y_test)


# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=0.01)
weights_path = None
model = LeNet.build(width=28, height=28, depth=1, classes=10, weightsPath=weights_path)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
 
# only train and evaluate the model if we *are not* loading a
# pre-existing model
if weights_path == None:
    print("[INFO] training...")
    model.fit(training_features, training_labels, batch_size=128, epochs=20,
        verbose=1)
 
    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(test_features, test_labels,
        batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
    
    model.save_weights('lenet_weights', overwrite=True)