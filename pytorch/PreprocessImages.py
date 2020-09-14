import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2

# MODIFY THIS WITH YOUR OWN PATHS
HOME = "$YOUR_HOME_PATH"
PNA_DATA_PATH = os.path.join(HOME, "$YOUR_DATA_PATH")
PNA_TRAIN_FOLDER = os.path.join(PNA_DATA_PATH, "stage_2_train_images/")
PNA_OUT_DIR = os.path.join(HOME, "$YOUR_OUTPUT_DIR")
PNA_LABELS = os.path.join(PNA_DATA_PATH, "stage_2_train_labels.csv")

# DEFINE CLASS DICTIONARY
CLASS_DICT = {0: "healthy", 1: "sick"}

# LOAD TRAIN LABELS
dflabels = pd.read_csv(PNA_LABELS)
df_patients = dflabels[["patientId"]]
df_targets = dflabels[["Target"]]
patients = df_patients.values.flatten()
targets = df_targets.values.flatten()

# LOAD TRAINING SPLITS
splits_url = "https://raw.githubusercontent.com/QTIM-Lab/Assessing-Saliency-Maps/master/pneumonia_splits.csv"
df = pd.read_csv(splits_url)
df_splits = df[["Split"]]
df_images = df[["Image_ID"]]
splits = df_splits.values.flatten()
images = df_images.values.flatten()

usplits = np.unique(splits)
dict = {usplit: images[np.where(splits == usplit)] for usplit in usplits}

# BUILD DATA FOLDERS
for usplit in dict:
    # PREPARE OUTPUT FOLDERS
    SPLIT_OUT_DIR = os.path.join(PNA_OUT_DIR, "%s" % usplit)
    if not os.path.exists(SPLIT_OUT_DIR):
        os.mkdir(SPLIT_OUT_DIR)
        for uclass in CLASS_DICT:
            TARGET_OUT_DIR = os.path.join(SPLIT_OUT_DIR, CLASS_DICT[uclass])
            os.mkdir(TARGET_OUT_DIR)
    split_imgs = dict[usplit]
    img_count = 0
    for img in split_imgs:
        img_id = re.search("(.*).png", img).group(1)
        img_path = os.path.join(PNA_TRAIN_FOLDER, "%s.dcm" % img_id)
        img_target = int(targets[np.where(patients == img_id)][0])
        img_out_path = os.path.join(
            SPLIT_OUT_DIR, "%s/%s" % (CLASS_DICT[img_target], img)
        )
        ds = pydicom.read_file(img_path)
        img = ds.pixel_array
        if cv2.imwrite(img_out_path, img):
            print(
                "%d/%d    PatientID: %s"
                % (img_count, len(split_imgs), ds.data_element("PatientID"))
            )
            img_count += 1
