from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2, l1
from keras.layers import Flatten, Dense, Dropout
from keras.utils import plot_model
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras.optimizers import Adam
from keras import backend as K
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


class CrackDetector(object):
    def __init__(self):
        self.model = None
        self.subimage_size = (50, 50)       # Size of the sub image which be classified
        self.conv_filters = [6, 12, 24]     # Number of filters in each layer
        self.conv_kernels = [3, 3, 3]       # kernel sizes of each conv layer
        self.conv_strides = [1, 1, 1]       # conv layer strides
        self.conv_padding = "valid"         # Padding "same" or "valid"
        self.conv_activation = 'elu'        # Activation for each Conv layer
        self.conv_regularization = 'l2'     # Regularization in each conv layer (None or 'l1', 'l2')
        self.use_maxpooling = True          # Max pooling to follow each conv layer?
        self.regularization_val = 0.01      # Activity regularizer penalty
        self.dense_layers = [100, 25, 10]   # After flattening, profile of dense layers
        self.dense_activations = ['elu', 'elu', 'elu']
        self.dense_dropouts = [0.5, 0.5, 0.5]
        self.learning_rate = 0.0001
        self.model_file = None

    def build_model(self):
        """Builds the convolutional neural network to detect cracks in images
        """

        # Initialize a sequential model
        model = Sequential()
        nlayer = 0

        # Add all convolutional layers
        for i in range(len(self.conv_filters)):
            conv_filter = self.conv_filters[i]
            kernel_size = self.conv_kernels[i]
            stride = self.conv_strides[i]

            # Regularization
            if self.conv_regularization == 'l1':
                regularizer = l1(self.regularization_val)
            elif self.conv_regularization == 'l2':
                regularizer = l2(self.regularization_val)
            else:
                regularizer = None



            if i == 0:
                input_shape = (self.subimage_size[0], self.subimage_size[0], 3)
                model.add(Conv2D(conv_filter, kernel_size,
                                 strides=(stride, stride),
                                 input_shape=input_shape,
                                 padding=self.conv_padding,
                                 kernel_regularizer=regularizer,
                                 activation=self.conv_activation))
            else:
                model.add(Conv2D(conv_filter, kernel_size,
                                 strides=(stride, stride),
                                 padding=self.conv_padding,
                                 kernel_regularizer=regularizer,
                                 activation=self.conv_activation))
            nlayer += 1
            # print("CONVOLUTION LAYER {} : {}".format(nlayer, model.layers[-1].output_shape))

            if self.use_maxpooling:
                model.add(MaxPooling2D())
                # print("MAXPOOLING  LAYER {} : {}".format(nlayer, model.layers[-1].output_shape))

        # Flatten
        model.add(Flatten())
        model.add(Dropout(self.dense_dropouts[0]))
        nlayer += 1
        # print("\nFLATTEN     LAYER {} : {}".format(nlayer, model.layers[-1].output_shape))

        # All all dense layers
        for i in range(len(self.dense_layers)):
            model.add(Dense(self.dense_layers[i],
                            activation=self.dense_activations[i]))
            # Dropout
            if i+1 < len(self.dense_dropouts):
                model.add(Dropout(self.dense_dropouts[i+1]))

            nlayer += 1
            # print("DENSE       LAYER {} : {}".format(nlayer, model.layers[-1].output_shape))

        # Finally let's add the output layer
        model.add(Dense(1, activation='softmax'))

        # Minimization
        adam_optimizer = Adam(lr=self.learning_rate)
        model.compile(adam_optimizer, 'mse')

        # Plot to file
        if self.model_file is not None:
            plot_model(model, to_file=self.model_file, show_shapes=True)

        # Print summary
        model.summary()

        return model


    def train_model(self, imgs_original, imgs_analyzed, picklefile='cdam.pkl'):
        """Performs training and saves weights"""
        # TODO
        pass

    def test_model(self, img):
        # TODO
        pass



if __name__ == '__main__':
    net = CrackDetector()
    net.build_model()