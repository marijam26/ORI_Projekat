import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, MaxPooling2D, \
    AveragePooling2D
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import batch_normalization_v1
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

from tensorflow import keras

warnings.filterwarnings('ignore')


def prepareImages(train, shape, path):
    x_train = np.zeros((shape, 100, 100, 3))
    count = 0

    for fig in train['Image']:

        # load images into images of size 100x100x3
        img = keras.utils.load_img(path + "/" + fig, target_size=(100, 100, 3))
        x = keras.utils.img_to_array(img)
        x = preprocess_input(x)

        x_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    return x_train



if __name__ == '__main__':
    print("Hello")
    train = pd.read_csv("train.csv")
    train.info()
    #print(train.shape)
    print(train.Id.describe())
    #print(train.head(10))

    y_train = train["Id"]
    X_train = train.drop(labels=["Id"], axis=1)
    print(y_train.head())
    print(train.isnull().sum().sum())

    x_train = prepareImages(train, train.shape[0], "train")
    x_train = x_train / 255.0
    print("x_train shape: ", x_train.shape)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_train = to_categorical(y_train, num_classes=4251)
    print(y_train.shape)
    print(y_train)

    mod = Sequential()

    mod.add(Conv2D(32, (7, 7), strides=(1, 1), name='conv0', input_shape=(100, 100, 3)))

    mod.add(BatchNormalization(axis=3, name='bn0'))
    mod.add(Activation('relu'))

    mod.add(MaxPooling2D((2, 2), name='max_pool'))
    mod.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
    mod.add(Activation('relu'))
    mod.add(AveragePooling2D((3, 3), name='avg_pool'))

    mod.add(Flatten())
    mod.add(Dense(500, activation="relu", name='rl'))
    mod.add(Dropout(0.8))
    mod.add(Dense(4251, activation='softmax', name='sm'))
    mod.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = mod.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1)
    gc.collect()

    # model = Sequential()
    #
    # model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(100, 100, 3)))
    # model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # fully connected
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(y_train.shape[1], activation="softmax"))
    # model.summary()
    #optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
    #learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                #patience=3,
                                                #verbose=1,
                                                #factor=0.5,
                                                #min_lr=0.00001)
    #model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
    #history = model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=2, callbacks=[learning_rate_reduction])