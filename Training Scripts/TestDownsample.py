import Utilities as utils
import numpy as np
import os

testPath = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Patient Data/"
data_path = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Patient Data/"
model_dir = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Models/"
filenames = os.listdir(data_path)

data = utils.LoadRaw(testPath,filenames)
