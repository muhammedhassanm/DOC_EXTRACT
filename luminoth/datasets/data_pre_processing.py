# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:48:44 2019

@author: 100119
"""

import pandas as pd
df = pd.read_csv('D:/DOC_EXTRACT/luminoth/datasets/val.csv', header=None)
df.rename(columns={0: 'image_id', 1: 'xmin',2:'ymin',3:'xmax',4:'ymax',5:'label'}, inplace=True)
df.to_csv('D:/DOC_EXTRACT/luminoth/test.csv', index=False) # save to new csv file
