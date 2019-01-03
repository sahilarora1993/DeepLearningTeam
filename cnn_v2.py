import requests
import zipfile
import pandas as pd
import numpy as np
import os
import nibabel
import matplotlib.pyplot as plt
import utils
import tensorflow as tf
from tensorflow.python.framework import ops
import shutil
import math

# data preprocessing

data_path1 = '/work/06023/tg853226/stampede2/oasis/OASIS3/OASIS3'
#read oasis_data
labels = pd.read_csv("oasis_label_2.csv")
# days since MRI
labels['Days since MRI'] = pd.to_numeric(labels['Label'].str[-4:])
# month and two month since MRI
labels['Months since MRI'] = np.floor(labels['Days since MRI']/30)
labels['2 months since MRI'] = np.floor(labels['Days since MRI']/60)
# drop subjects who have two mri at the same day 
new = labels
new = new.drop_duplicates(subset=['Subject','2 months since MRI'])
# sort by subject and then months since mri
sorted_labels = new.sort_values(by=['Subject','Months since MRI'])

# need to change format since MRI format is OAS30073_MR_d3670.nii, 
# but nameID on excel file format is OAS30073_ClinicalData_d3670
# OAS30001_ClinicalData_d0000 -> OAS30001_MR_d0000.nii


def rep(s):
    x = s.replace("_ClinicalData_","_MR_")
    return x
sorted_labels['New Label'] = sorted_labels['Label'].apply(rep) + '.nii'

# divide sorted_label into 3 label-normal, uncertain and AD_labels
normal_labels = sorted_labels[sorted_labels['dx1'] == 'Cognitively normal']
uncertain_labels = sorted_labels[sorted_labels['dx1'] == 'uncertain dementia']
AD_labels = sorted_labels[sorted_labels['dx1'] == 'AD Dementia']


sorted_labels['New Label'] = sorted_labels['Label'].apply(rep) + '.nii'
######################################################################################################################

# oasis_image_data_dates.csv matches sorted_labels
image_data = pd.read_csv("oasis_image_data_dates.csv")


def match(df, col1 = 'subject',col2 = 'days'):
    df1=sorted_labels[sorted_labels["Subject"]==df[col1]]
    loc = (np.abs(df1["Days since MRI"] - df[col2])).argmin()            
    low_limit = df[col2] - 180
    high_limit = df[col2] + 180
    if sorted_labels.loc[loc]['Days since MRI'] < high_limit and sorted_labels.loc[loc]['Days since MRI'] > low_limit:
#         #here return dx1 other than the nearest day
#         return int(sorted_labels.loc[loc]['Days since MRI'])
        return sorted_labels.loc[loc]['dx1']

    else:
        return None
image_data["match_label"] = image_data.apply(match,col1 = 'subject',col2 = 'days',axis = 1)
image_data.head()


# def match(day_file,subj_file,excel):
#     df1=excel[excel["Subject"]==subj_file]
#     loc = (np.abs(df1["Days since MRI"] - day_file)).argmin()            
#     low_limit = day_file - 180
#     high_limit = day_file + 180
#     if excel.loc[loc]['Days since MRI'] < high_limit and excel.loc[loc]['Days since MRI'] > low_limit:
#         return excel.loc[loc]['Days since MRI']
#     else:
#         return None
# df['nearest'] = df.apply(lambda x: match(x['days'],x['subject'],sorted_labels),axis=1) 
# df.head()

#get sample file name for AD NORMAL AND UNCERTAIN

sample_normal = image_data[image_data['match_label'] == 'Cognitively normal']
normal_file_name = sample_normal['actual file name'].values

sample_AD = image_data[image_data['match_label'] == 'AD Dementia']
AD_file_name = sample_AD['actual file name'].values

sample_uncertain = image_data[image_data['match_label'] == 'uncertain dementia']
uncertain_file_name = sample_uncertain['actual file name'].values

#print("sample normal", len(sample_normal))
#print("sample AD", len(sample_AD))
#print("sample uncertain", len(sample_uncertain))

files = os.listdir(data_path1)

#read data
X = np.zeros((1571,256,256,256))
Y = np.zeros((1571,1))
i = 0
# 0 is normal 1 is ad and 2 is uncertain
for data_file in files:
    for normal_name in normal_file_name:
        if data_file == normal_name:
            X[i] = nibabel.load(data_path + data_file).get_data()
            Y[i] = 0
            i += 1
    for AD_name in AD_file_name:
        if data_file == AD_name:
            X[i] = nibabel.load(data_path + data_file).get_data()
            Y[i] = 1
            i += 1
    for uncertain_name in uncertain_file_name:
        if data_file == uncertain_name:
            X[i] = nibabel.load(data_path + data_file).get_data()
            Y[i] = 2
            i += 1
    if i == 150:
             
        break

# split into train and test

indices = np.random.permutation(X.shape[0])
indices

training_idx, test_idx = indices[:3], indices[3:]
Y = pd.get_dummies(Y.squeeze()).values


X_train, X_test = X[training_idx], X[test_idx]
Y_train, Y_test = Y[training_idx,:], Y[test_idx,:]

y_test_cls = Y_test.squeeze()
y_train_cls = Y_train.squeeze()
y_test_cls.shape


print(X_train.shape ,X_test.shape ,Y_train.shape,Y_test.shape,y_test_cls.shape,y_train_cls.shape)