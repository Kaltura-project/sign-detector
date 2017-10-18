#!/usr/bin/env python3

from multiprocessing import Process, Queue
from queue import PriorityQueue
import cv2
import numpy as np

CAPTURE_SOURCE = "test_footage.mp4"
sign_cascade = cv2.CascadeClassifier("classifier/cascade.xml")


def detection_worker(w_input, w_output):
    try:
        while True:
            frame = w_input.get()
            if isinstance(frame, type(None)):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            signs = sign_cascade.detectMultiScale(gray, 1.1, 5)

            ret = [len(signs)]
            for sign in signs:
                x_pos, y_pos, width, height = sign
                ret.append(((x_pos, y_pos), (x_pos + width, y_pos + height)))
            print(len(signs))
            w_output.put(ret)

    except KeyboardInterrupt:
        pass


def sign_detect():
    try:
        worker_input = Queue()
        worker_output = Queue()

        detector = Process(target=detection_worker, args=(
            worker_input, worker_output))
        detector.start()

        cap = cv2.VideoCapture(CAPTURE_SOURCE)
        dim = None

        while cap.isOpened():
            ret, frame = cap.read()
            worker_input.put(frame)

            if not worker_output.empty():
                dim = worker_output.get()
                print(dim)
                for x in range(dim[0]):
                    cv2.rectangle(frame, dim[x + 1][0], dim[x + 1][1], (255, 0, 0), 4)

            cv2.imshow("frame", frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        worker_input.put(None)
        detector.join()
        cap.release()

if __name__ == "__main__":
    sign_detect()
