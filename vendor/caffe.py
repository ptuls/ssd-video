# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import caffe

from core.constants import RESULT_LAYER_NAME


def load_transformer(net_shape, mean_array):
    # input pre-processing
    transformer = caffe.io.Transformer({'data': net_shape})
    # change from H by W by C to C by H by W
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean_array)

    # set to BGR order first
    transformer.set_channel_swap('data', (2, 1, 0))
    return transformer


def setup_device(is_gpu_enabled, gpu_id=0):
    if is_gpu_enabled:
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        logging.info('Using GPU id %d', gpu_id)
        return True
    else:
        caffe.set_mode_cpu()
        logging.info('No GPU found, using CPU')
    return False


def load_net(model_definition, model_weights):
    """
    Load the neural network and set the dimensions for batching.

    Arguments:
        model_definition: the definition of the model found in the prototext
        model_weights: weights of the neural network model
        batch_size: integer to set the batch size
    """
    return caffe.Net(model_definition, model_weights, caffe.TEST)


def reshape_net(net, batch_size):
    input_width = net.blobs['data'].width
    input_height = net.blobs['data'].height
    num_channels = net.blobs['data'].channels

    net.blobs['data'].reshape(batch_size, num_channels, input_width, input_height)


def detect(net, image_transformed, layer_name=RESULT_LAYER_NAME):
    reshape_net(net, 1)
    net.blobs['data'].data[...] = image_transformed
    return net.forward()[layer_name]


def get_input_geometry(net):
    """
    Get input dimension of neural network.

    Arguments:
        net: neural network model
    """
    return net.blobs['data'].data.shape
