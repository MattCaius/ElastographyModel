#Imports required for function
from scipy.io import loadmat
import pandas as pd
import numpy as np
import random as rand
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pathlib
import json

def preprocessData(data):
    height = data.shape[0] #get height of raw data
    width =  data.shape[1] #get width of raw data
    lowerH = int((height-1651)/2) #find lower bound to position cropped section in center of frame
    upperH = lowerH+1651 #set upper height bound
    lowerW = int((width-205)/2) #find lower bound to position cropped section in center of frame
    upperW = lowerW + 205 #set upper width bound

    frames = [] #create list to hold frames
    for x in range(data.shape[2]): #range is how many frames there are in the set
        frames.append(data[lowerH:upperH,lowerW:upperW,x]) #cutdata to match size expected by model

    pairs = generatePairs(frames) #gerenate pairs of frames
    for x in range(len(pairs)): #convert all arrays to tensor's to match models input
        pairs[x][2] = tf.convert_to_tensor(pairs[x][2], np.float32)
    return pairs

def predict(processed):

    predictions = [] #create list to hold predicitons
    print(len(processed)) #print number of frame pairs to bench mark progress against and ensure proper pair genaration
    for x in range(len(processed)):
        #predict pair label
        probability = model.predict(np.expand_dims(np.expand_dims(processed[x][2], axis=3),axis=0))[0]
        if (x%50==0): #print x every fifty as way to show progress through set
            print(x)
        if (probability>0.5): #if probabilty above set target label 1 for optimal pair
            predictions.append([1,probability])
        else: #otherwise label 0 for bad pair
            predictions.append([0,probability])
    return predictions

def formatOutput(processed,predictions):
    output = [] #list to hold output values
    for x in range(len(predictions)):
        #add frame 1, frame 2 and probability of being a good pair to output
        output.append({'frame1':processed[x][0],'frame2':processed[x][1],'percent':round(float(predictions[x][1][0]),3)})
    output.sort(key=lambda a: a['percent'],reverse=True) #sort output to find best pairs
    return output

def generatePairs(frames):
    framePairs = [] #vector to hold image pairs

    for x in range(len(frames)): #x represents the first frame in the pair
        for y in range(len(frames)-3-x): #y represents second frame in the pair
            pair = [(x+1),(x+y+4),np.stack((frames[x], frames[x+y+3]), axis=-1)] #generate pair
            framePairs.append(pair)
    return framePairs

def get_model(width, height, depth):

    inputs = keras.Input((width, height, depth, 1))
    
    x = layers.Conv3D(filters=64, kernel_size=(3,3,2), activation="relu")(inputs) #Convolutional layer which is automatically trained to detect desired features
    x = layers.MaxPool3D(pool_size=(2,2,1))(x) #Pooling layer which reduces the dimensionality of the RF data to allow for faster processing in further layers
    x = layers.BatchNormalization()(x) #Normalizes the data post convolution and pooling to allow for faster training

    x = layers.Conv3D(filters=64, kernel_size=(3,3,1), activation="relu")(x) #Convolutional layer which is automatically trained to detect desired features
    x = layers.MaxPool3D(pool_size=(2,2,1))(x) #Pooling layer which reduces the dimensionality of the RF data to allow for faster processing in further layers
    x = layers.BatchNormalization()(x) #Normalizes the data post convolution and pooling to allow for faster training

    x = layers.Conv3D(filters=128, kernel_size=(3,3,1), activation="relu")(x) #Convolutional layer which is automatically trained to detect desired features
    x = layers.MaxPool3D(pool_size=(2,2,1))(x) #Pooling layer which reduces the dimensionality of the RF data to allow for faster processing in further layers
    x = layers.BatchNormalization()(x) #Normalizes the data post convolution and pooling to allow for faster training

    x = layers.Conv3D(filters=256, kernel_size=(3,3,1), activation="relu")(x) #Convolutional layer which is automatically trained to detect desired features
    x = layers.MaxPool3D(pool_size=(2,2,1))(x) #Pooling layer which reduces the dimensionality of the RF data to allow for faster processing in further layers
    x = layers.BatchNormalization()(x) #Normalizes the data post convolution and pooling to allow for faster training

    x = layers.Conv3D(filters=512, kernel_size=(3,3,1), activation="relu")(x) #Convolutional layer which is automatically trained to detect desired features
    x = layers.MaxPool3D(pool_size=(2,2,1))(x) #Pooling layer which reduces the dimensionality of the RF data to allow for faster processing in further layers
    x = layers.BatchNormalization()(x) #Normalizes the data post convolution and pooling to allow for faster training

    x = layers.GlobalAveragePooling3D()(x) #Calculates average output of each feature map in previous layers in preperation for final classification
    x = layers.Dense(units=512, activation="relu")(x) #Neurons in layer are connected to each previous neuron, performs matrix-vector multiplication
    x = layers.Dropout(0.3)(x) #Helps to prevent overfitting

    outputs = layers.Dense(units=1, activation="sigmoid")(x) #defining final neuron for classification

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

#Set up model and load weights
model = get_model(width=1651, height=205, depth=2)
model.load_weights(pathlib.Path(__file__).parent / "3d_image_classification.h5")

def predictfile(filename):
    #Read in .mat raw data file
    print(pathlib.Path(__file__).parent.parent/"GUI"/"uploads"/filename)
    rawMatFile = loadmat(pathlib.Path(__file__).parent.parent/"GUI"/"uploads"/filename) 
    processfile = rawMatFile['rf1']
    #process raw data to make predictions one
    processed = preprocessData(processfile)
    #predict labeles for processed data
    predictions = predict(processed)
    #format output
    output = formatOutput(processed,predictions)
    os.remove(pathlib.Path(__file__).parent.parent/"GUI"/"uploads"/filename)
    return output

