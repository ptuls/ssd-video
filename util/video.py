# -*- coding: utf-8 -*-
from threading import Thread
import cv2

from Queue import Queue


class VideoStream(object):
    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        self.queue = Queue(maxsize=queue_size)

    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.queue.full():
                grabbed, frame = self.stream.read()

                if not grabbed:
                    self.stop()
                    return

                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True

    def more(self):
        return self.queue.qsize() > 0
