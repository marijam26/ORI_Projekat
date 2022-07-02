import csv
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import warnings
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def prepare_images(train, shape, path):
    x_train = np.zeros((shape, 100, 100, 3))
    count = 0

    for fig in train['Image']:

        # load images into images of size 100x100x3
        img = keras.utils.load_img(path + "/" + fig, target_size=(100, 100, 3))
        x = keras.utils.img_to_array(img)
        x = preprocess_input(x)

        x_train[count] = x
        if count % 500 == 0:
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    return x_train


def remove_ids(unique_ids):
    input = open('test.csv', 'r')
    output = open('results.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(output)
    for row in csv.reader(input):
        if row[1] in unique_ids:
            writer.writerow(row)
    input.close()
    output.close()


if __name__ == '__main__':
    train = pd.read_csv("train_all.csv")
    train.info()
    test = pd.read_csv("test.csv")
    # test.info()
    # print(train.shape)
    # train, test = train_test_split(train, test_size=0.2)

    y = train["Id"]
    X = train.drop(labels=["Id"], axis=1)
    print(train.Id.describe())
    unique_id = np.unique(train.Id.array)
    print(unique_id.size)

    x_train = prepare_images(train, train.shape[0], "train")
    x_train = x_train / 255.0
    print("x_train shape: ", x_train.shape)

    # za testiranje pomocu rucno napravljenog skupa
    # x_test = prepareImages(test, test.shape[0], "train")
    # x_test = x_test / 255.0
    # print("x_train shape: ", x_train.shape)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y)
    y_train = to_categorical(y_train, num_classes=unique_id.size)
    print("y_train:", y_train)

    # encoder gresi kad se ovo doda
    # y_test = label_encoder.fit_transform(test["Id"])
    # y_test = to_categorical(y_test, num_classes=unique_id.size)
    # print("Drugi:", y_test)

    # mod = keras.models.load_model("MyModel3")

    mod = Sequential()

    mod.add(Conv2D(16, (7, 7), strides=(1, 1), name='conv0', input_shape=(100, 100, 3)))
    mod.add(BatchNormalization(axis=3, name='bn0'))
    mod.add(Activation('relu'))
    mod.add(MaxPooling2D((2, 2), name='max_pool'))

    mod.add(Conv2D(32, (3, 3), strides=(1, 1), name="conv1"))
    mod.add(Activation('relu'))
    mod.add(AveragePooling2D((3, 3), name='avg_pool'))

    mod.add(Flatten())
    mod.add(Dense(500, activation="relu", name='rl'))
    mod.add(Dropout(0.8))
    mod.add(Dense(y_train.shape[1], activation='softmax', name='sm'))
    mod.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = mod.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1)
    mod.save("MyModel4")
    gc.collect()

    plt.plot(history.history['accuracy'], color='g', label="Train Accuracy")
    plt.title("Train Accuracy")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # za testiranje pomocu rucno napravljenog skupa
    # predictions = mod.predict(np.array(x_test), verbose=1)
    # hits = 0

    test1 = os.listdir("test")
    col = ['Image']
    test_data = pd.DataFrame(test1, columns=col)
    test_data['Id'] = ''
    x_test1 = prepare_images(test_data, test_data.shape[0], "test")
    x_test1 /= 255
    predictions = mod.predict(np.array(x_test1), verbose=1)

    for i, pred in enumerate(predictions):
        test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
        # predict = label_encoder.inverse_transform(pred.argsort()[-5:][::-1])
        # whale_id = test.loc[i, "Id"]
        # print(predict)
        # print(whale_id)
        # if whale_id in predict:
        #     hits = hits + 1

    test_data.to_csv('submission.csv', index=False)

    # za testiranje pomocu rucno napravljenog skupa
    # print("Hits: ", hits)
    # print("Test accuracy: ", hits / test.shape[0])
