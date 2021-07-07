import Utilities as utils
from Custom_Classes import RF_PairLoader
from sklearn.model_selection import train_test_split
import Models
import tensorflow.keras as keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# import all relevant data

filenames =[
    "P32-W1-S4.mat",
    "P39-W2-S4.mat",
    "P39-W4-S6.mat",
    "P87-W0-S3.mat",
    "P87-W2-S4.mat",
    "P90-W0-S4.mat",
    "P94-W1-S3.mat",
]

data_path = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Patient Data/"
model_dir = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Models/"
directory = utils.LoadDir(data_path, "Data_Directory.csv")
rawData = utils.LoadRaw(data_path, filenames)

train_dir, valid_dir = train_test_split(directory, test_size= 0.2)
train_dir = train_dir.reset_index()
valid_dir = valid_dir.reset_index()

train_loader = RF_PairLoader(train_dir, rawData, scan_col="ScanID", i_col="Frame 1",
                             j_col="Frame 2", labels = "Label axial", batch_size=4)

valid_loader = RF_PairLoader(valid_dir, rawData, scan_col="ScanID", i_col="Frame 1",
                             j_col="Frame 2", labels = "Label axial", batch_size=4)

hyperparams = {
    'dropout_rate' : 0.3,
    'initial_LR' : 0.0001,
    'decay_rate' : 0.96
}

model = Models.Get_3dCNN(train_loader.__getitem__(0)[0].shape, hyperparameters=hyperparams)

already_trained = True
if not already_trained:
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_dir + "3d_image_classification.h5", save_best_only=True
    )

    # Train the model, doing validation at the end of each epoch
    epochs = 50
    model.fit(
        train_loader,
        validation_data=valid_loader,
        epochs=epochs,
        verbose=1,
        callbacks= [early_stopping_cb, checkpoint_cb]
    )

y_true = valid_dir["Label axial"]

model.load_weights(model_dir + "3d_image_classification.h5")

NPVs, PPVs = utils.PPV_NPV_analysis(model, valid_loader, y_true[:252], 0.95, 0.5)