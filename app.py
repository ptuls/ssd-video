# -*- coding: utf-8 -*-
import argparse
import time
import sys
import logging
import platform
import cv2

from core.common import get_imagenet_mean, get_voc_labels
from core.process_frame import process_frame
from util.video import VideoStream
from vendor.caffe import detect, get_input_geometry, load_net, load_transformer, setup_device


FRAME_SIZE = (640, 480)
CONFIDENCE_THRESHOLD = 0.3


def run():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=True, help='path to input video file')
    ap.add_argument('-o', '--output', required=True, help='path to output video file')
    ap.add_argument('-d', '--definition', required=True, help='path to model definition')
    ap.add_argument('-w', '--weights', required=True, help='path to model weights')
    args = vars(ap.parse_args())

    logging.basicConfig(stream=sys.stdout)
    logging.info('starting video file thread...')

    model_definition = args['definition']
    model_weights = args['weights']
    logging.info('setting up GPU and neural network...')

    setup_device(False)
    try:
        net = load_net(model_definition, model_weights)
    except RuntimeError as err:
        logging.error(err)
        sys.exit()

    try:
        transformer = load_transformer(get_input_geometry(net), get_imagenet_mean())
    # unfortunately Transformer can throw a general exception; see caffe source
    except Exception as err:
        logging.error(err)
        sys.exit()

    stream = VideoStream(args['video']).start()

    try:
        platform.mac_ver()
        try:
            fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
        except AttributeError:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args['output'], fourcc, 25, FRAME_SIZE)
    except:
        try:
            fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
        except AttributeError:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args['output'], fourcc, 25, FRAME_SIZE)
    time.sleep(1.0)

    # loop over frames from the video file stream
    logging.info('processing frames...')
    while stream.more():
        frame = stream.read()
        frame_resized = cv2.resize(frame, FRAME_SIZE)
        detections = detect(net, transformer.preprocess('data', frame_resized))
        frame_detected = process_frame(detections, frame_resized, get_voc_labels(), CONFIDENCE_THRESHOLD)
        out.write(frame_detected)


if __name__ == '__main__':
    run()
