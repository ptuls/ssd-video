# -*- coding: utf-8 -*-
import cv2
from core.constants import CLASS_COLORS


def draw_bounding_boxes(img, results):
    img_cp = img.copy()
    for result in results:
        object_class = result['class']
        confidence = result['confidence']
        xMin = int(result['boundingBox']['xMin'])
        yMin = int(result['boundingBox']['yMin'])
        xMax = int(result['boundingBox']['xMax'])
        yMax = int(result['boundingBox']['yMax'])

        # flip around to BGR for opencv
        red, green, blue = CLASS_COLORS[object_class]
        box_color = (blue, green, red)

        cv2.rectangle(
            img_cp,
            (xMin, yMin),
            (xMax, yMax),
            box_color,
            2
        )

        box_text = ''.join([object_class, ': ', '%.2f' % confidence])
        ret, baseline = cv2.getTextSize(box_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)

        cv2.rectangle(
            img_cp,
            (xMin, yMax - ret[1] - baseline),
            (xMin + ret[0], yMax),
            box_color,
            -1
        )
        cv2.putText(
            img_cp,
            box_text,
            (xMin, yMax - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255, 0),
            1
        )
    return img_cp
