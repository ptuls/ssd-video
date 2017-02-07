# -*- coding: utf-8 -*-
import logging


def detect_objects(
    detections,
    image_size,
    label_names,
    confidence_threshold
):
    """
    Detect objects for a single image. Can be used with different neural network models.

    Arguments:
        detections_per_image: a 6-tuple with (class, confidence, xmin, ymin, xmax, ymax)
        image_size: size of the original image
        label_names: names of the classes the neural network was trained with
        confidence_threshold: only return results for objects with confidence above this threshold
    """
    results = []

    detection_label = detections[0, 0, :, 1]
    detection_conf = detections[0, 0, :, 2]
    detection_x_min = detections[0, 0, :, 3]
    detection_y_min = detections[0, 0, :, 4]
    detection_x_max = detections[0, 0, :, 5]
    detection_y_max = detections[0, 0, :, 6]

    # only output boxes with confidence above threshold
    top_indices = [i for i, conf in enumerate(detection_conf) if conf >= confidence_threshold]

    top_confidence = detection_conf[top_indices]
    top_label_indices = detection_label[top_indices].tolist()
    top_x_min = detection_x_min[top_indices]
    top_y_min = detection_y_min[top_indices]
    top_x_max = detection_x_max[top_indices]
    top_y_max = detection_y_max[top_indices]

    for i in xrange(top_confidence.shape[0]):
        img_height, img_width = image_size

        x_min = int(round(top_x_min[i] * img_width))
        y_min = int(round(top_y_min[i] * img_height))
        x_max = int(round(top_x_max[i] * img_width))
        y_max = int(round(top_y_max[i] * img_height))
        conf = top_confidence[i]
        label = label_names[int(round(top_label_indices[i]))]

        results.append({
            'class': label,
            'boundingBox': {
                'xMin': x_min,
                'yMin': y_min,
                'xMax': x_max,
                'yMax': y_max,
            },
            'confidence': float(conf),
        })

        logging.debug(
            '%s: %s [%d, %d, %d, %d], Confidence: %.3f',
            label,
            x_min,
            y_min,
            x_max,
            y_max,
            conf,
        )

    return results
