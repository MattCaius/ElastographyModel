import pandas as pd

# THIS IS A TEST FOR GIT


#read in the label datasets

scan_IDs = [
    'P39-W2-S4',
    'P39-W4-S6',
    'P87-W0-S3',
    'P87-W2-S4',
    'P90-W0-S4',
    'P94-W1-S3'
]

data_dir = "/home/matthew/Desktop/AI/Biomechanics Lab/Elastography Frame-Pair Evaluator/Patient Data/"
file = data_dir + "Labels.xlsx"

sheets = list()

for scan_ID in scan_IDs:
    sheet = pd.read_excel(file, sheet_name=scan_ID)[['Frame 1', 'Frame 2', 'Label axial']]
    sheet["ScanID"] = scan_ID + ".mat"
    sheets.append(sheet)

directory = pd.concat(sheets)
directory = directory.drop(directory[directory['Label axial'] == 0.5].index)


# supersample the bad pairs

good_pairs = directory[directory["Label axial"] == 1]
bad_pairs = directory[directory["Label axial"] == 0]

new_good_pairs = good_pairs.sample(len(bad_pairs)-len(good_pairs), replace=True)
good_pairs = pd.concat([good_pairs,new_good_pairs])

directory = pd.concat([good_pairs,bad_pairs])


#shuffle and save

directory = directory.sample(frac = 1)
directory.to_csv(data_dir + "Data_Directory.csv", index = False)

