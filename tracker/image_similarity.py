# import the necessary packages
import os
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
# import the necessary packages
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import concatenate
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.applications import resnet
from keras.layers import AveragePooling2D
# specify the shape of the inputs for our network
IMG_SHAPE = (64, 64, 1)
target_shape = (64, 64)
# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 10

# define the path to the base output directory
BASE_OUTPUT = "output2"
# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# 準備數據
left_train_images_path = "left_train"
right_train_images_path = "right_train"
left_test_images_path = "left_test"
right_test_images_path = "right_test"
trainX = []
trainY = []
testX = []
testY = []

class image_similarity_model(object):
    def __init__(self):
        imgA = Input(shape=IMG_SHAPE)
        imgB = Input(shape=IMG_SHAPE)
        featureExtractor = self.build_siamese_model(IMG_SHAPE)
        featsA = featureExtractor(imgA)
        featsB = featureExtractor(imgB)
        merge_layer = Lambda(self.euclidean_distance, output_shape=(1,))([featsA, featsB])
        normal_layer = BatchNormalization()(merge_layer)
        output_layer = Dense(1, activation="sigmoid")(normal_layer)
        self.model = Model(inputs=[imgA, imgB], outputs=output_layer)
        self.model.compile(loss=self.loss(margin=1), optimizer="RMSprop",
	    metrics=["accuracy"])
        self.model.load_weights("image_similarity_3.h5")
        print(self.model.summary())
        
    def preprocess_image(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        return image

    # Model
    def build_siamese_model(self, inputShape, embeddingDim=48):
        inputs = Input(inputShape)
        x = BatchNormalization()(inputs)
        x = Conv2D(4, (5, 5), activation="tanh")(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Conv2D(16, (5, 5), activation="tanh")(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)

        x = BatchNormalization()(x)
        x = Dense(10, activation="tanh")(x)
        model = Model(inputs, x)
        return model

    def loss(self, margin=1):
        def contrastive_loss(y_true, y_pred):
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - (y_pred), 0))
            return K.mean((1 - y_true) * square_pred + (y_true) * margin_square)
        return contrastive_loss
    
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def predict(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, target_shape, interpolation=cv2.INTER_AREA)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2, target_shape, interpolation=cv2.INTER_AREA)
        
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        img1 = np.expand_dims(img1, axis=-1)
        img2 = np.expand_dims(img2, axis=-1)
        
        img1 = np.reshape(img1, (1, 64, 64))
        img2 = np.reshape(img2, (1, 64, 64))
        acc = self.model.predict([img1, img2])
        return acc
    