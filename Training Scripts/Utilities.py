import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import random

# Downsample the RF Data into a resolution the network is trained on
# input raw data

def CropToShapeRaw(Data, target_length):

    lowerI = int((Data['rf1'].shape[0]-target_length) / 2)

    if lowerI < 0:
        print("This scan is far too small")
        return None

    Data['rf1'] = tf.slice(Data['rf1'],[lowerI, 0, 0], [target_length, 256, Data['rf1'].shape[2]])

    return Data

# Generate the pair between ith and jth frame and return it as tensor, for use in custom generator
# Returns shape [1, i, j , 2]

def GenFramePair(i, j, frameData, scan):

    pairs = tf.convert_to_tensor(np.stack((frameData[scan]['rf1'][:,:,i - 1],
                                          frameData[scan]['rf1'][:,:,j - 1]), axis=-1))
    return tf.expand_dims(pairs, axis = 0)

# Generate a plot to describe the negative and positive predictive value
# Return NPV/PPV at various thresholds

def PPV_NPV_analysis(model, testloader, y_true, PPV_target, NPV_target, save = False, path = None):

    y_pred = model.predict(testloader)[:,0]

    NPVs = list()
    PPVs = list()

    for thresh in np.linspace(0,1,20):
        preds = np.copy(y_pred)
        preds[preds >= thresh] = 1
        preds[preds < thresh] = 0

        tp = np.sum((preds == 1) & (y_true[:y_pred.shape[0]] == 1))
        tn = np.sum((preds != 1) & (y_true[:y_pred.shape[0]] != 1))
        fp = np.sum((preds == 1) & (y_true[:y_pred.shape[0]] != 1))
        fn = np.sum((preds != 1) & (y_true[:y_pred.shape[0]] == 1))

        NPV = tn/(tn+fn)
        PPV = tp/(tp+fp)

        if (NPV>=NPV_target) & (PPV>=PPV_target):
            print(NPV,PPV, thresh)

        NPVs.append(NPV)
        PPVs.append(PPV)

    plt.scatter(np.nan_to_num(NPVs),np.nan_to_num(PPVs))
    plt.axvline(NPV_target)
    plt.hlines(PPV_target,0,1)
    plt.xlabel("NPV")
    plt.ylabel("PPV")

    if save:
        plt.savefig(path+"PPV_NPV.png")

    plt.show()

    return NPVs,PPVs

# ROC Curve Generate and Plot
# Also AUC

def ROC_Analysis(model, testloader, y_true, save = False, path = None):

    y_pred = model.predict(testloader)[:,0]

    FPR, TPR, Thresholds = roc_curve(y_true[:y_pred.shape[0]], y_pred)
    AUC = roc_auc_score(y_true[:y_pred.shape[0]],y_pred)

    plt.plot(FPR, TPR)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve, AUC: " + str(AUC))

    if save:
        plt.savefig(path+"ROC.png")

    plt.show()

# Load in the raw data
# Outputs a dict with structure dict[scanID]

def LoadRaw(path, filenames):

    print("Loading Data...")

    RawData = dict()

    for file in filenames:
        if file.endswith(".mat"):
            RawData[file] = CropToShapeRaw(loadmat(path + file), 2062)

    print("The Data is Loaded")

    return RawData

# Load in the data directory

def LoadDir(path, filename):
    return pd.read_csv(path+filename)

# train/test split but add a stratification element

def Stratify_Split(Directory, proportion, column = None):

    # Can use as normal train test split
    if column is None:
        train_dir, valid_dir = train_test_split(Directory, test_size=proportion)
        train_dir = train_dir.reset_index()
        valid_dir = valid_dir.reset_index()
        return train_dir, valid_dir

    # Find uniques in column of interest
    uniques = set(Directory[column])
    num_unique = len(uniques)
    n = int(proportion*num_unique) # no to hold out

    print("holding out", n, "of " + column)

    validation = random.sample(uniques, n)
    training = uniques.difference(validation)

    valid_dir = Directory.loc[Directory[column].isin(validation)].reset_index()
    train_dir = Directory.loc[Directory[column].isin(training)].reset_index()

    return train_dir, valid_dir

# Pad bndr files to match

def PadBndr(BndrData, target):

    pad_i = target[0] - BndrData.shape[0]
    pad_j = target[1] - BndrData.shape[1]

    paddings = tf.constant([[0,pad_i], [0,pad_j]])

    if pad_i != 0 or pad_j != 0:
        BndrData = tf.pad(BndrData, paddings, "CONSTANT")

    return tf.expand_dims(BndrData, -1)

# load boundary files

def LoadBndrData(path, filenames):

    print("Loading Boundaries")

    BndrData = dict()

    for file in filenames:
        if file.endswith(".mat"):

            bndrname = file.replace(".mat","-T.mat")

            try:
                BndrData[bndrname] = PadBndr(np.asarray(loadmat(path + bndrname)["TumorArea"]), [1984,226])
            except FileNotFoundError:
                print("This file has no corresponding tumor boundary, which may cause problems if it is in the training or test set")

    print("Done Loading Boundaries")
    return BndrData

# Get specific boundary

def GetBndr(data, filename):

    return tf.expand_dims(data[filename], 0)