# this file conatins functions to build the various models

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from Custom_Classes import RandomTranslator
from tensorflow.keras.models import Sequential
from Custom_Classes import Normalized_Correlation_Layer

def Get_3dCNN(input_shape, hyperparameters,batch_size, compile = True):

    inputs = layers.Input((input_shape[1], input_shape[2], input_shape[3], input_shape[4]))

    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 2), activation="relu")(inputs)  # Convolutional layer
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

    if compile:
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

def Get_3dCNN_skinny(input_shape, hyperparameters, batch_size):

    inputs = layers.Input((input_shape[1], input_shape[2], input_shape[3], input_shape[4]))

    transformed = RandomTranslator(batch_size)(inputs)

    x = layers.Conv3D(filters=16, kernel_size=(3, 3, 2), activation="relu")(transformed)  # Convolutional layer
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)  # Pooling layer
    # x = layers.BatchNormalization()(x)  # Normalizes

    x = layers.Conv3D(filters=16, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=16, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=16, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)


    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 1), activation="relu")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
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

def Get_XCorr(input_shape, hyperparameters, batch_size, compile = True):

    inputs = layers.Input(input_shape[1:])

    x = RandomTranslator(batch_size)(inputs)

    a = keras.backend.expand_dims(x[:,:,:,0], axis= -1)
    b = keras.backend.expand_dims(x[:,:,:,1], axis= -1)

    model = Sequential()
    model.add(layers.Conv2D(kernel_size=(5, 5), filters=64,
                            input_shape=(x.shape[1], x.shape[2], 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=64, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=128, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(kernel_size=(3, 3), filters=256, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    feat_map1 = model(b)
    feat_map2 = model(a)

    normalized = Normalized_Correlation_Layer(stride = (1,1), patch_size= (5,5))([feat_map1, feat_map2])

    normalized = tf.concat(normalized, -1)

    x = layers.Conv2D(kernel_size=(1, 1), filters=128, activation='relu')(normalized)
    x = layers.Conv2D(kernel_size=(3, 3), filters=256, activation=None)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Conv2D(kernel_size=(3, 3), filters=256, activation=None)(x)
    features = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(512)(features)
    x = layers.Dropout(hyperparameters["dropout_rate"])(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    if compile:

        model = keras.Model(inputs=inputs, outputs=output, name="XCorrNet")

        initial_learning_rate = hyperparameters["initial_LR"]
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate= hyperparameters['decay_rate'], staircase=True
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["acc"]
        )
    else:
        model = keras.Model(inputs=inputs, outputs=features)

    return model

def Get_BndrCNN(bndrshape):

    inputs = layers.Input((bndrshape[1],bndrshape[2],bndrshape[3]))

    x = layers.Conv2D(16,3,activation="relu", padding="same")(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(16,3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32,3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64,3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(128,3, activation="relu", padding="same")(x)

    out = layers.GlobalAvgPool2D()(x)

    Model = keras.Model(inputs=inputs, outputs = out)

    return Model

def GetMixedModel(input_shapes, hyperparameters, batch_size):

    CNN = Get_XCorr(input_shapes[0], hyperparameters, batch_size, compile=False)

    BndrCNN = Get_BndrCNN(input_shapes[1])

    x = layers.concatenate(inputs= [CNN.output, BndrCNN.output])

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(hyperparameters["dropout_rate"])(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(hyperparameters["dropout_rate"])(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs = [CNN.input, BndrCNN.input], outputs = output, name="XCorrNet")

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
