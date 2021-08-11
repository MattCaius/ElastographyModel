import Utilities as utils
from Custom_Classes import RF_PairLoader
from sklearn.model_selection import train_test_split
import Models
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

batch_size = 8

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# import all relevant data

label_path = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Labeling/"
data_path = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Patient Data/"
model_dir = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Models/Str"
filenames = os.listdir(data_path)

directory = utils.LoadDir(label_path, "Data_Directory.csv")
rawData = utils.LoadRaw(data_path, filenames)
# bndrFile = utils.LoadBndrData(data_path + "TB/", filenames)

train_dir, valid_dir = utils.Stratify_Split(directory, 0.4, "ScanID")

valid_dir, test_dir = utils.Stratify_Split(valid_dir, 0.5, "ScanID")

train_loader = RF_PairLoader(train_dir, rawData, scan_col="ScanID", i_col="Frame 1",
                             j_col="Frame 2", labels = "Label axial", bndr_col="TB",
                             bndrData=bndrFile, batch_size=batch_size, is3D=False)

valid_loader = RF_PairLoader(valid_dir, rawData, scan_col="ScanID", i_col="Frame 1",
                             j_col="Frame 2", labels = "Label axial", bndr_col="TB",
                             bndrData=bndrFile, batch_size=batch_size,  is3D=False)

test_loader = RF_PairLoader(test_dir, rawData, scan_col="ScanID", i_col="Frame 1",
                             j_col="Frame 2", labels = "Label axial", bndr_col="TB",
                             bndrData=bndrFile, batch_size=batch_size, is3D=False)

hyperparams = {
    'dropout_rate' : 0.6,
    'initial_LR' : 0.0001,
    'decay_rate' : 0.96
}

# input_shapes = [train_loader.__getitem__(0)[0][0].shape, train_loader.__getitem__(0)[0][1].shape]

model = Models.Get_XCorr(train_loader.__getitem__(0)[0].shape, hyperparameters=hyperparams, batch_size= batch_size)

print("got model")
model.summary()

wait = input()

already_trained = False

if not already_trained:
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_dir + "3d_image_classification_Mixed.h5", save_best_only=True
    )

    # Train the model, doing validation at the end of each epoch
    epochs = 100
    model.fit(
        train_loader,
        validation_data=valid_loader,
        epochs=epochs,
        verbose=1,
        callbacks= [early_stopping_cb, checkpoint_cb]
    )

y_true = test_dir["Label axial"]

model.load_weights(model_dir + "3d_image_classification_Mixed.h5")

utils.ROC_Analysis(model, test_loader, y_true, save = True, path = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Models/Str")

