# -*- coding: utf-8 -*-
import numpy as np


def get_voc_labels():
    """Labels for VOC PASCAL trained models."""
    return [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    ]


def get_imagenet_mean():
    """Mean for centering images trained on ImageNet datasets."""
    return np.array([103.939, 116.779, 128.68])


def get_imagenet_labels(conf):
    """All 1000 classes in the ImageNet contest."""
    with open(conf.get('IMAGENET_CLASSES_PATH'), 'rU') as image_classes:
        return image_classes.read().splitlines()
