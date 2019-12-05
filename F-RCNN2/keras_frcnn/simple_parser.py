import cv2,os
import numpy as np
import pandas as pd

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

#    visualise = True
    
#    with open(input_path,'r') as f:
    
    df = pd.read_table(input_path, delimiter=',', names=('filename','x1','y1','x2','y2','class_name'))
#    df = pd.read_csv('/content/F-RCNN/keras-frcnn-master/annotate.csv')
    df['x1'] = df['x1'].astype(int)
    df['x1'] = df['y1'].astype(int)
    df['x1'] = df['x2'].astype(int)
    df['x1'] = df['y2'].astype(int)
    for _, row in df.iterrows():
#    with open(input_path) as f:
#        print('Parsing annotation files')
#        data = f.readlines()
#        for line in data:
#            line_split = line.strip().split(',')
        print(row)
        class_name_ = row['class_name']
        filename_ = row['filename']
        x1_ = row['x1']
        y1_ = row['y1']
        x2_ = row['x2']
        y2_ = row['y2']
        
#            (filename,x1,y1,x2,y2,class_name) = line_split
        (filename,x1,y1,x2,y2,class_name) = filename_,x1_,y1_,x2_,y2_,class_name_
#            print(filename)

        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if class_name not in class_mapping:
            if class_name == 'bg' and found_bg == False:
                print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                found_bg = True
            class_mapping[class_name] = len(class_mapping)

        if filename not in all_imgs:
            
            all_imgs[filename] = {}
            filename = filename.replace('"',"")
            img = cv2.imread(filename)
       
            (rows,cols) = img.shape[:2]
            all_imgs[filename]['filepath'] = filename
            all_imgs[filename]['width'] = cols
            all_imgs[filename]['height'] = rows
            all_imgs[filename]['bboxes'] = []
            if np.random.randint(0,6) > 0:
                all_imgs[filename]['imageset'] = 'trainval'
            else:
                all_imgs[filename]['imageset'] = 'test'

        all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(y1), 'y1': int(x2), 'y2': int(y2)})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])
        
        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
        
        return all_data, classes_count, class_mapping
