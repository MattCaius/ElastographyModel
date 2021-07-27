import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# Downsample the RF Data into a resolution the network is trained on
# input raw data

def CropToShapeRaw(Data, target_length):

    lowerI = int((Data['rf1'].shape[0]-target_length) / 2)

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

    RawData = dict()

    for file in filenames:
        if file.endswith(".mat"):
            RawData[file] = CropToShapeRaw(loadmat(path + file),2062)

    return RawData

# Load in the data directory

def LoadDir(path, filename):
    return pd.read_csv(path+filename)