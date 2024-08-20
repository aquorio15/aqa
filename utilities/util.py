from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from munch import munchify
import sys
import csv
import base64
import time
import pandas as pd
import numpy as np

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    if 'pvqa' in fname:
        return pvqa_load_tsv(fname)
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


"""
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
pvqa_FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def pvqa_load_tsv(fname: str):
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    df = pd.read_csv(fname, delimiter='\t', names=pvqa_FIELDNAMES)

    split = None
    for s in ('train', 'test', 'val'):
        if s in fname:
            split = s

    data = []
    for i in range(df.shape[0]):
        datum = {}
        datum['img_id'] = '%s_%04d' % (split, df['image_id'][i])
        datum['img_w'] = df['image_w'][i]
        datum['img_h'] = df['image_h'][i]
        datum['num_boxes'] = df['num_boxes'][i]

        boxes = df['boxes'][i]
        buf = base64.b64decode(boxes[1:])
        temp = np.frombuffer(buf, dtype=np.float64).astype(np.float32)
        datum['boxes'] = temp.reshape(datum['num_boxes'], -1)

        features = df['features'][i]
        buf = base64.b64decode(features[1:])
        temp = np.frombuffer(buf, dtype=np.float32)
        datum['features'] = temp.reshape(datum['num_boxes'], -1)

        data.append(datum)
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self):
        self.mean = [0.43216, 0.394666, 0.37645]
        self.std = [0.22803, 0.22145, 0.216989]
        tfs = []
        tfs.append(transforms.Resize((112, 112)))
        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR('RGB'=='BGR'))
        tfs.append(ToRange255(max([0, 1])==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class LoadTransformImage(object):

    def __init__(self):
        self.load = LoadImage()
        self.tf = TransformImage()

    def __call__(self, path_img):
        img = self.load(path_img)
        tensor = self.tf(img)
        return tensor


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x