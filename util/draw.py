# -*- coding: utf-8 -*-
import cv2


def draw_bounding_boxes(img, results):
    img_cp = img.copy()
    for result in results:
        object_class = result['class']
        confidence = result['confidence']
        xMin = int(result['boundingBox']['xMin'])
        yMin = int(result['boundingBox']['yMin'])
        xMax = int(result['boundingBox']['xMax'])
        yMax = int(result['boundingBox']['yMax'])
        cv2.rectangle(
            img_cp,
            (xMin, yMin),
            (xMax, yMax),
            (0, 0, 204),
            2
        )

        box_text = ''.join([object_class, ': ', '%.2f' % confidence])
        ret, baseline = cv2.getTextSize(box_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)

        cv2.rectangle(
            img_cp,
            (xMin, yMax - ret[1] - baseline),
            (xMin + ret[0], yMax),
            (0, 0, 204),
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
