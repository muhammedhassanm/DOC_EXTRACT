# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:19:01 2019

@author: 100119
"""

import os
import sys
import cv2
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import matplotlib.patches as patches
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# Root directory of the project
ROOT_DIR = os.path.abspath('D:/DOC_EXTRACT/Mask_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version



class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "table"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
#    NUM_CLASSES = 1 + 80 # COCO has 80 classes
    NUM_CLASSES = 1 + 1


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH ='D:/DOC_EXTRACT/Mask_RCNN/M-rcnn/Dataset_not_resized/mask_rcnn_epoch_50_not_resized_data.1575449978.709749.h5'




config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=COCO_MODEL_PATH, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG',"table"]
#IMAGE_DIR = os.path.join(ROOT_DIR, "New folder")
# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
path ='D:/DOC_EXTRACT/Mask_RCNN/M-rcnn/Dataset_not_resized/train_data/00012.jpg'
image = cv2.imread(path) 
basename = os.path.splitext(os.path.basename(path))[0]
# Run detection
results = model.detect([image], verbose=0)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
#segmentation


def draw_image_with_boxes(path, boxes_list):
 # load the image
 data = pyplot.imread(path)
 # plot the image
 pyplot.imshow(data)
 # get the context for drawing boxes
 ax = pyplot.gca()
 # plot each box
 for box in boxes_list:
      # get coordinates
      y1, x1, y2, x2 = box
      # calculate width and height of the box
      width, height = x2 - x1, y2 - y1
      # create the shape
      rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=5)
      # draw the box
      ax.add_patch(rect)
 # show the plot
 pyplot.show()  
 
draw_image_with_boxes(path, results[0]['rois'])
 
def get_width(xy):
    width = abs(xy[1] - xy[3])
    return width

def get_height(xy):
    height = abs(xy[0] - xy[2])
    return height

def get_area(xy):
    width = get_width(xy)
    height = get_height(xy)
    area = width * height
    return area

def get_biggest_box(xy_list):
#    biggest_area = 0
    boxes=[]
    index =[]
    for i, xy in enumerate(xy_list):
        print(xy)
        boxes.append(xy)
        index.append(i)
    
#        area = get_area(xy)
#        if area > biggest_area:
#        biggest_area = area
#        biggest_xy = xy
#        ix = i
#    return biggest_xy, ix
    return boxes,index

def overlay_box(image, xy): 
    position = (xy[1], xy[0])
    width = get_width(xy)
    height = get_height(xy)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle(position, 
                             width, 
                             height,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    plt.show()
    

boxes, indices = get_biggest_box(r['rois'])


def make_box_mask(image, xy):    
    target = image[xy[0]:xy[2], xy[1]:xy[3], :]
    img = np.zeros_like(image)
    img[xy[0]:xy[2], xy[1]:xy[3], :] = target
    plt.imshow(img)
    
    return img
for box in boxes:
    
    overlay_box(image, box)
    make_box_mask(image, box)

def make_segmentation_mask(image, mask):
    img = image.copy()
    img[:,:,0] *= mask
    img[:,:,1] *= mask
    img[:,:,2] *= mask
#    cv2.imwrite('C:/Users/100119/Desktop/Mask_Rcnn_Rebuild/Mask_RCNN-2.1/test muzzle/test muzzle/cropped/'+ basename + ".jpg",img)
    plt.imshow(img)
    return img

for ix in indices:
    mask = r['masks'][:,:,ix]
    mask = r['masks'][:,:,1]
    make_segmentation_mask(image, mask)

    mask.shape


       

