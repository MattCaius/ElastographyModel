# this file conatins functions to build the various models

import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from Custom_Classes import RandomTranslator
from keras import metrics


def Get_3dCNN(input_shape, hyperparameters):

    inputs = layers.Input((input_shape[1], input_shape[2], input_shape[3], input_shape[4]))

    transformed = RandomTranslator()(inputs)

    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 2), activation="relu")(transformed)  # Convolutional layer
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)  # Pooling layer
    # x = layers.BatchNormalization()(x)  # Normalizes

    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=512, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(hyperparameters["dropout_rate"])(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")

    initial_learning_rate = hyperparameters["initial_LR"]
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate= hyperparameters['decay_rate'], staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"]
    )

    return model