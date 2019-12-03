import glob
import cv2
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
count = 1
for image in glob.glob('D:/DOC_EXTRACT/Mask_RCNN/M-rcnn/Dataset_not_resized/train_data/*.jpg'):
     print(image)
     image=cv2.imread(image)   
#     image = cv2.resize(image,(600,1024),fx=2.5,fy=2.5)
     plt.imshow(image)
     plt.show()
     #Save Each Transformation
     cv2.imwrite('D:/DOC_EXTRACT/Mask_RCNN/M-rcnn/Dataset_not_resized/000' +  str(count)+ '.jpg',image)
#     os.rename(image,'D:/DOC_EXTRACT/train/doc_res_' +str(count)+ '.jpg')
     count=count+1
     
     
#foo = Image.open('C:/Users/100119/Desktop/DATA_EXTRACTION_DOCUMENT/table/table_images/0147_090.png')
#foo = foo.resize((300,300),Image.ANTIALIAS)
#foo.save("C:/Users/100119/Desktop/DATA_EXTRACTION_DOCUMENT/0147_090.jpg",quality=95)
#foo.save("C:/Users/100119/Desktop/DATA_EXTRACTION_DOCUMENT/0147_090.jpg",optimize=True,quality=95)
