# -*- coding: utf-8 -*-

"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import pickle
from PIL import Image, ImageOps
from io import BytesIO
import requests
import random
import sys

def load_data():
    # Load the data set
    hotdog = pickle.load(open("hotdog.pickle", "rb"))
    X = hotdog['X']
    #Y = [ 0 if x != 1 else 1 for x in hotdog['Y'] ]
    Y = hotdog['Y']
    X_test = hotdog['X_test']
    #Y_test = [ 0 if x != 1 else 1 for x in hotdog['Y_test'] ]
    Y_test = hotdog['Y_test']

    # Shuffle the data
    X, Y = shuffle(X, Y)
    Y = tflearn.data_utils.to_categorical(Y, nb_classes=101)
    Y_test = tflearn.data_utils.to_categorical(Y_test, nb_classes=101)
    return X,Y,X_test,Y_test








def train():
    X,Y,X_test,Y_test = load_data()
    # Train it! We'll do 100 training passes and monitor it as it goes.
    model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='hotdog-classifier')

    # Save model when training is complete to a file
    model.save("hotdog-classifier.tfl")
    print("Network trained and saved as hotdog-classifier.tfl!")

def load():
    # Make sure the data is normalized
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping, rotating and blurring the
    # images on our data set.
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    # Define our network architecture:

    # Input is a 32x32 image with 3 color channels (red, green and blue)
    network = input_data(shape=[None, 32, 32, 3],
                        data_preprocessing=img_prep,
                        data_augmentation=img_aug)

    # Step 1: Convolution
    network = conv_2d(network, 32, 3, activation='relu')

    # Step 2: Max pooling
    network = max_pool_2d(network, 2)

    # Step 3: Convolution again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 4: Convolution yet again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 5: Max pooling again
    network = max_pool_2d(network, 2)

    # Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.5)

    # Step 8: Fully-connected neural network with two outputs (0=isn't a hotdog, 1=is a hotdog) to make the final prediction
    network = fully_connected(network, 101, activation='softmax')

    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)
    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='hotdog-classifier.tfl.ckpt')
    model.load('hotdog-classifier.tfl')
    return model

model = load()

def boxed(img):
    img.thumbnail((32, 32))
    bbox = img.getbbox()
    locations = [
        (0,0,bbox[2],bbox[3]),
        (32-bbox[2],32-bbox[3],32,32),
        (0,32-bbox[3],bbox[2],32),
        (32-bbox[2],0,32,bbox[3])
    ]
    thumb = Image.new('RGB', (32, 32), (255, 255, 255))
    thumb.paste(img, locations[0])
    return thumb

def containsHotdog(urls, auth):
    for url in urls:
        result, maxClass, classes = isHotdog(url, auth)
        if result:
            return True

    return False

def isHotdog(url, auth=None):
    
    r = requests.get(url) if auth is None else requests.get(url, headers = {'Authorization': 'Bearer ' + auth})
    img = Image.open(BytesIO(r.content))
    img = boxed(img)
    data = np.array(img.getdata())/255.0
    data.resize(img.height, img.width, 3)
    print("running predict", file=sys.stderr)
    predictions = model.predict(np.array([data]))
    print("predict done", file=sys.stderr)
    maxClass = np.argmax(np.array(predictions).flatten())
    
    return True if (maxClass == 100) else False, maxClass.item(), np.array(predictions).flatten()