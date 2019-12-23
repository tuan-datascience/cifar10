import keras
from keras.datasets import cifar10
from keras import Model
from keras import layers, optimizers
import cv2
import numpy as np
from keras.utils import np_utils

num_classes = 10

(img_train, label_train), (img_test, label_test) = cifar10.load_data()
img_train = np.array([cv2.resize(img_train[i], (128, 128)) for i in range(0, img_train.shape[0])])
img_test = np.array([cv2.resize(img_test[i], (128, 128)) for i in range(0, img_test.shape[0])])
img_train = 1.0 * img_train / 255.0
img_test = 1.0 * img_test / 255.0

label_train = np_utils.to_categorical(label_train, num_classes)
label_test = np_utils.to_categorical(label_test, num_classes)


def inception_modulde(x,
                      filters_1x1,
                      filters_3x3_reduce,
                      filters_3x3,
                      filters_5x5_reduce,
                      filters_5x5,
                      filters_pool_proj,
                      name=None
                      ):
    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                             bias_initializer=bias_init)(x)

    conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu',
                             kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                             bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu',
                             kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
                             bias_initializer=bias_init)(conv_5x5)

    pool_proj = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',
                              kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output


kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', kernel_initializer=kernel_init,
                  bias_initializer=bias_init)(input_layer)
x = layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
x = layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)
x = layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

x = layers.BatchNormalization()(x)
x = inception_modulde(x, 64, 96, 128, 16, 32, 32)
x = inception_modulde(x, 128, 128, 192, 32, 96, 64)

x = layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

x = layers.BatchNormalization()(x)
x = inception_modulde(x, 192, 96, 208, 16, 48, 64)
x = inception_modulde(x, 240, 128, 256, 32, 128, 64)
x = inception_modulde(x, 256, 156, 280, 64, 156, 128)

x1 = layers.AveragePooling2D((5, 5), strides=3)(x)

x1 = layers.BatchNormalization()(x1)
x1 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)

x1 = layers.BatchNormalization()(x1)
x1 = layers.Flatten()(x1)
x1 = layers.Dense(1024, activation='relu')(x1)
x1 = layers.Dropout(0.3)(x1)
x1 = layers.Dense(10, activation='softmax')(x1)

model = Model(input_layer, x1, name='inception_v1')
model.summary()

sgd = optimizers.SGD(lr=0.02,momentum=0.9,nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
history = model.fit(x=img_train,y=label_train,epochs=150,batch_size=256,validation_data=(img_test,label_test))
model.save('cifar10_model.h5')