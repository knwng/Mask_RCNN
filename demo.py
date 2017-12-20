import os
import sys
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.io

import coco
import utils
import model as modellib
import visualize
import cv2

# %matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = '/home/qwang/origin_hd'
OUTPUT_DIR = '/home/qwang/mask_hd'

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 18
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 1024
    # IMAGE_PADDING = False
    MASK_SHAPE = [28, 49]

BATCH_SIZE = 18
config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
print('totally {} file to proecee'.format(len(file_names)))
ITERS = math.ceil(float(len(file_names)/BATCH_SIZE))
for i in range(ITERS):
    print('Iteration: {}, processing image {}-{}'.format(i, i*BATCH_SIZE+1, (i+1)*BATCH_SIZE))
    image = []
    for j in range(BATCH_SIZE):
        if(i*BATCH_SIZE+j >= len(file_names)):
            break
        image.append(skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i*BATCH_SIZE+j])))

    # Run detection
    results = model.detect(image, verbose=1)
    
    # Visualize results
    # r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])
    # print('result shape {}'.format(len(results)))
    for k in range(len(results)):
        r = results[k]
        inst_num = (np.array(r['class_ids']).shape)[0]
        idx = [x for x in range(inst_num) if (r['class_ids'][x] == 1 and r['scores'][x] > 0.99) ]
        score_idx = zip([r['scores'][x] for x in idx], idx)
        score_idx = sorted(score_idx, key=lambda x:x[0])
        # print('idx for person: {}'.format(idx))
        # print('score for person: {}'.format([r['scores'][x] for x in idx ]))
        # print('class for person: {}'.format([class_names[r['class_ids'][x]] for x in idx]))

        for j in range(min(len(idx), 2)):
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'mask_{}_{}.jpg'.format(file_names[i*BATCH_SIZE+k].split('.jpg')[0], j)), np.array(r['masks'][:, :, score_idx[-j-1][1]]) * 255)
    # if i==0:
    #     break

