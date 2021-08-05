from scipy.io import loadmat
import pandas as pd
import os
import numpy as np

labelPath = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Labeling/"

frame1 = list()
frame2 = list()
label = list()
srcfiles = list()

for root, dirs, files in os.walk(labelPath):
    print(root)
    for file in files:
        if "disp.mat" in file:
            srcfile = file.replace("-disp","")
            labels = loadmat(labelPath + file)
            frame1.extend(labels["DispQuality"][:,0])
            frame2.extend(labels["DispQuality"][:,1])
            label.extend(labels["DispQuality"][:,2])
            srcs = [srcfile]*labels["DispQuality"][:,2].shape[0]
            srcfiles.extend(srcs)

    directory = pd.DataFrame(list(zip(frame1,frame2,label,srcfiles)), columns=["Frame 1","Frame 2","Label axial", "ScanID"])

    good_pairs = directory[directory["Label axial"] == 1]
    bad_pairs = directory[directory["Label axial"] == 0]

    if len(good_pairs) < len(bad_pairs):
        new_good_pairs = good_pairs.sample(len(bad_pairs) - len(good_pairs), replace=True)
        good_pairs = pd.concat([good_pairs, new_good_pairs])
    else:
        new_bad_pairs = bad_pairs.sample(len(good_pairs) - len(bad_pairs), replace=True)
        bad_pairs = pd.concat([bad_pairs, new_bad_pairs])

    directory = pd.concat([good_pairs, bad_pairs])

    # shuffle and save

    directory = directory.sample(frac=1)
    directory.to_csv(labelPath + "Data_Directory_Disp.csv", index=False)
