import os
import sys
import random
import math
import imageio
import requests
from io import BytesIO

import numpy as np
import visualize

import coco
import tensorflow as tf
import utils
import model as modellib


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCnn:
    _class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    DEVICE = "/cpu:0"
    DATA_DIR = os.path.join(ROOT_DIR, "data")

    def __init__(self):
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
        config = InferenceConfig()
        config.display()
        with tf.device(self.DEVICE):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=config)
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        self.biggest_object = 0
        self.object_area = 0
        self.output = {}

    def get_object_area(self, p):
        return np.sum(np.array(self.mask, dtype=np.uint8)) / p


    def __get_objects(self, url, test=False, local=False):
        if not test:
            response = requests.get(url)
            img = imageio.imread(BytesIO(response.content))

        else:
            img = imageio.imread(url)
        p = (img.shape[0] * img.shape[1]) / 100
        self.results = self.model.detect([img], verbose=1)[0]

        if not(self.results['class_ids'].any()):
            return None
        self.masked_img = visualize.display_instances(np.array(img), self.results['rois'], self.results['masks'],
                                             self.results['class_ids'], self._class_names, self.results['scores'])
        for i, class_id in zip(range(len(self.results['class_ids'])), self.results['class_ids']):
            self.mask = self.results['masks'][:, :, i]

            if self._class_names[class_id] in self.output.keys():
                if class_id == 1:
                    self.object_area = self.get_object_area(p)
                    if self.biggest_object < self.object_area:
                        self.biggest_object = self.object_area
                    self.output.append([self.results['rois'][i],
                                        self.results['masks'][:, :, i],
                                        self.results['scores'][i]])
        else:
            if class_id == 1:
                obj_area = get_object_area(results['masks'][:, :, i], p)
                biggest_object = obj_area if biggest_object < obj_area else biggest_object
            out[class_names[class_id]] = [results['rois'][i],
                                          results['masks'][:, :, i],
                                          results['scores'][i]
                                          ]
    out['masked_img'] = masked_img
    out['class_instance_dic'] = class_instance_dic

    return
