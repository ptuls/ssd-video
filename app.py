# -*- coding: utf-8 -*-
import argparse
import time
import sys
import logging
import platform
import cv2

from util.video import VideoStream

FRAME_SIZE = (640, 480)


def run():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=True, help='path to input video file')
    ap.add_argument('-o', '--output', required=True, help='path to output video file')
    args = vars(ap.parse_args())

    logging.basicConfig(stream=sys.stdout)
    logging.info('starting video file thread...')
    stream = VideoStream(args['video']).start()

    try:
        platform.mac_ver()
        out = cv2.VideoWriter(args['output'], cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), 25, FRAME_SIZE)
    except:
        out = cv2.VideoWriter(args['output'], cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 25, FRAME_SIZE)
    time.sleep(1.0)

    # loop over frames from the video file stream
    while stream.more():
        frame = stream.read()
        frame_resized = cv2.resize(frame, FRAME_SIZE)
        out.write(frame_resized)


if __name__ == '__main__':
    run()
