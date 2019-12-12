# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:23:03 2019

@author: 100119
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

os.chdir('C:/Users/100119/Desktop/TABLE_EXTRACT/Datasets')

TRAIN_CSV_PATH = os.path.join(os.getcwd(),'train.csv')
TEST_CSV_PATH =  os.path.join(os.getcwd(),'test.csv')

train_df = pd.read_csv(TRAIN_CSV_PATH)


train_df.columns = ['filename','xmin','ymin','xmax','ymax','class']
train_df = train_df[['filename','class','xmin','xmax','ymin','ymax']]
train_df.head()

test_df =  pd.read_csv(TEST_CSV_PATH)


test_df.columns = ['filename','xmin','ymin','xmax','ymax','class']
test_df = test_df[['filename','class','xmin','xmax','ymin','ymax']]
test_df.head()

train_df['filename'].nunique()
train_df['class'].value_counts()

#New DataFrame

data = pd.DataFrame()
data['format'] = train_df['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    print(i)
    data['format'][i] = 'C:/Users/100119/Desktop/TABLE_EXTRACT/Datasets/train/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train_df['xmin'][i]) + ',' + str(train_df['ymin'][i]) + ',' + str(train_df['xmax'][i]) + ',' + str(train_df['ymax'][i]) + ',' + train_df['class'][i]

data.to_csv('C:/Users/100119/Desktop/TABLE_EXTRACT/Datasets/annotate.txt', header=None, index=None, sep=' ')
