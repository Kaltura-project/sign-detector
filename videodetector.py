#!/usr/bin/env python3

from multiprocessing import Process, Queue
from queue import PriorityQueue
import cv2
import numpy as np
# import time

CAPTURE_SOURCE = "./test_footage3.mp4"
sign_cascade = cv2.CascadeClassifier("classifier3/cascade.xml")


def sign_detect():
    try:
        cap = cv2.VideoCapture(CAPTURE_SOURCE)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output3.avi', fourcc, 30, (1280, 720))

        dim = None
        counter = 0

        myrec = []

        while cap.isOpened():
            counter += 1
            ret, frame = cap.read()
            h, w, l = frame.shape
            # frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if(counter % 10 == 0):
                myrec = []
            # start = Time.time()
            # print("Begin detection cycle: ", Time.time())
            if(counter % 5 == 0):
                signs = sign_cascade.detectMultiScale(gray, 1.1, 5)
                signarea = 0
                msign = -1
                for sign in range(len(signs)):
                    print("SIGN DETECTED: ", signs[sign])
                    area = signs[sign][2] * signs[sign][3]
                    print("ROI: ", area)
                    if(area > signarea):
                        signarea = area
                        msign = sign
                if(msign != -1):
                    print("EIGENSIGN: ", signs[msign])
                    x_pos, y_pos, width, height = signs[msign]
                    square_x = x_pos + int((width - height) / 2)
                    myrec = [(square_x, y_pos), (square_x +
                                                 height, y_pos + height)]
                # print("Detection cycle over: ", Time.time())
            if(myrec != []):
                cv2.rectangle(frame, myrec[0], myrec[1], (255, 0, 0), 4)

            out.write(frame)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        out.release()
        cap.release()

if __name__ == "__main__":
    sign_detect()
