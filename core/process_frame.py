# -*- coding: utf-8 -*-
from core.parse_results import detect_objects
from util.draw import draw_bounding_boxes


def process_frame(
    detections,
    frame,
    label_names,
    confidence_threshold
):
    height, width, _ = frame.shape
    results = detect_objects(detections, (height, width), label_names, confidence_threshold)
    new_frame = draw_bounding_boxes(frame, results)
    return new_frame
