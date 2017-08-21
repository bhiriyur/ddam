from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2, l1
from keras.layers import Flatten, Dense, Dropout
from keras.utils import plot_model
import glob
# from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras.optimizers import Adam
import pickle
# import os
# import cv2
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import sys
# from tqdm import tqdm


def extract_cracks(sub_img):
    nx, ny, _ = sub_img.shape
    ntotal = float(nx*ny)
    ncrack = 0
    min_threshold = 50
    max_threshold = 200
    for i in range(nx):
        for j in range(ny):
            if sub_img[i, j, 1] < min_threshold and  \
                            sub_img[i, j, 2] >= max_threshold and \
                            sub_img[i, j, 3] < min_threshold:
                ncrack += 1

    return float(ncrack)/ntotal


def apply_random_transform(img):
    # TODO
    return img


class CrackDetector(object):
    def __init__(self):
        # ConvNet Parameters
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

        # Training Parameters
        self.batch_size = 256
        self.nb_epochs = 10
        self.samples_per_epoch = 10

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

        self.model = model

    def data_generator(self, imgs_original, imgs_analyzed):

        while True:
            n = len(imgs_analyzed)
            x, y = [], []
            i = 0       # image number
            xbeg = 0
            ybeg = 0
            count = 0   # Sample number
            while count < self.batch_size:
                img_orig = plt.imread(imgs_analyzed[i])
                img_anal = plt.imread(imgs_analyzed[i])
                nx, ny, _ = img_orig.shape
                xend = xbeg + self.subimage_size[0]
                yend = ybeg + self.subimage_size[1]

                # Sub image and crack extent
                xi = img_orig[xbeg:xend, ybeg:yend, :]
                yi = extract_cracks(img_anal[xbeg:xend, ybeg:yend, :])

                # Add random transformations (flip, rotate, etc)
                xi_transformed = apply_random_transform(xi)

                x.append(xi_transformed)
                y.append(yi)
                count += 1

                # Move to next in same row
                xbeg += self.subimage_size[0]

                if xbeg + self.subimage_size[0] > nx:
                    # If reached end of row, move to next column and start from 0
                    ybeg += self.subimage_size[1]
                    xbeg = 0

                    if ybeg + self.subimage_size[1] > ny:
                        # If reached end of column also, move to next image and start over
                        i = (i+1) % n
                        xbeg = 0
                        ybeg = 0

            yield np.array(x), np.array(y)

    def train_model(self, imgs_original, imgs_analyzed, picklefile='cdam.h5'):
        """Performs training and saves weights"""
        if self.model is None:
            # Building model
            self.build_model()

        tgen = self.data_generator(imgs_original, imgs_analyzed)
        self.model.fit_generator(tgen, steps_per_epoch=self.samples_per_epoch,
                                 epochs=self.nb_epochs,
                                 verbose=1)

        if picklefile is not None:
            self.model.save(picklefile)
        pass

    def test_model(self, img):
        # TODO
        pass


if __name__ == '__main__':
    net = CrackDetector()
    net.build_model()
    train_images_orig = glob.glob("NineSigma\\images-for-training\\*original.jpg")
    train_images_anal = glob.glob("NineSigma\\images-for-training\\*analyzed_pixated.jpg")
    net.train_model(train_images_orig, train_images_anal)