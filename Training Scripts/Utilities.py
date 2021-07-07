import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

# Generate the pair between ith and jth frame and return it as tensor, for use in custom generator
# Returns shape [1, i, j , 2]

def GenFramePair(i, j, frameData, scan):
    pairs = tf.convert_to_tensor(np.stack((frameData[scan]['rf1'][:,:,i - 1],
                                          frameData[scan]['rf1'][:,:,j - 1]), axis=-1))
    return tf.expand_dims(pairs, axis = 0)

# Generate a plot to describe the negative and positive predictive value
# Return NPV/PPV at various thresholds

def PPV_NPV_analysis(model, testloader, y_true, PPV_target, NPV_target):

    y_pred = model.predict(testloader)[:,0]

    NPVs = list()
    PPVs = list()

    for thresh in np.linspace(0,1,20):
        preds = np.copy(y_pred)
        preds[preds >= thresh] = 1
        preds[preds < thresh] = 0

        tp = np.sum((preds == 1) & (y_true == 1))
        tn = np.sum((preds != 1) & (y_true != 1))
        fp = np.sum((preds == 1) & (y_true != 1))
        fn = np.sum((preds != 1) & (y_true == 1))

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
    plt.show()

    return NPVs,PPVs

# Load in the raw data
# Outputs a dict with structure dict[scanID]

def LoadRaw(path, filenames):

    RawData = dict()

    for file in filenames:
        RawData[file] = loadmat(path + file)

    return RawData

# Load in the data directory

def LoadDir(path, filename):
    return pd.read_csv(path+filename)