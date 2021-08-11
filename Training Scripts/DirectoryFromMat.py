from scipy.io import loadmat
import pandas as pd
import os
import numpy as np

labelPath = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Labeling/"
TB = labelPath + "TB/"

frame1 = list()
frame2 = list()
label = list()
srcfiles = list()
bndrfiles = list()

for root, dirs, files in os.walk(labelPath):
    print(root)
    for file in files:
        if "str.mat" in file:
            srcfile = file.replace("-str","")
            bndrfile = srcfile.replace(".mat", "-T.mat")
            labels = loadmat(labelPath + file)
            frame1.extend(labels["AxStrQuality"][:,0])
            frame2.extend(labels["AxStrQuality"][:,1])
            label.extend(labels["AxStrQuality"][:,2])
            srcs = [srcfile]*labels["AxStrQuality"][:,2].shape[0]
            srcfiles.extend(srcs)
            bndrs = [bndrfile]*labels["AxStrQuality"][:,2].shape[0]
            bndrfiles.extend(bndrs)

    directory = pd.DataFrame(list(zip(frame1,frame2,label,srcfiles,bndrfiles)), columns=["Frame 1","Frame 2","Label axial", "ScanID", "TB"])

    good_pairs = directory[directory["Label axial"] == 1]
    bad_pairs = directory[directory["Label axial"] == 0]

    new_good_pairs = good_pairs.sample(len(bad_pairs) - len(good_pairs), replace=True)
    good_pairs = pd.concat([good_pairs, new_good_pairs])

    directory = pd.concat([good_pairs, bad_pairs])

    # shuffle and save

    directory = directory.sample(frac=1)
    directory.to_csv(labelPath + "Data_Directory.csv", index=False)
