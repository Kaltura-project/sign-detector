#!/usr/bin/env python3

from multiprocessing import Process, Queue
from queue import PriorityQueue
import cv2
import numpy as np

CAPTURE_SOURCE = "test_footage.mp4"
numrec = 1
sign_cascade = cv2.CascadeClassifier("classifier3/cascade.xml")


def sign_detect():
    try:
        cap = cv2.VideoCapture(CAPTURE_SOURCE)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 30, (1280, 720))
        dim = None

        while cap.isOpened():
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            signs = sign_cascade.detectMultiScale(gray, 1.1, 5)

            q = PriorityQueue()

            for sign in signs:
                x_pos, y_pos, width, height = sign
                area = width * height
                q.put((-area, (x_pos, y_pos), (x_pos + width, y_pos + height)))

            numdisp = numrec

            if(len(signs) < numrec):
                numdisp = len(signs)

            for x in range(numdisp):
                rec = q.get()
                cv2.rectangle(frame, rec[1], rec[2], (255, 0, 0), 4)

            out.write(frame)
            # cv2.imshow("frame", frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        cap.release()
        out.release()

if __name__ == "__main__":
    sign_detect()
