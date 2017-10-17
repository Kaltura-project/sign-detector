import cv2
import numpy as np
import os

start = 101
end = 201


def extract_noise():
    filecounter = 0
    folderstring = "StopSignDataset/negatives/"
    for filenum in range(start, end):
        filestring = folderstring + str(filenum) + ".jpg"
        frame = cv2.imread(filestring)
        print(filestring, np.shape(frame))
        w = int(max(np.shape(frame)) / 10)
        # w2 = int(w / 3)
        # w -= w2
        # while(w >= w2):
        for x in range(0, frame.shape[0], w):
            for y in range(0, frame.shape[1], w):
                if x + w <= frame.shape[1] and y + w <= frame.shape[0]:
                    roi = frame[y:y + w, x:x + w]
                    cv2.imwrite("StopSignDataset/extra/" + str(filecounter) + ".jpg", roi)
                    filecounter += 1
            # w -= w2
    print(0, filecounter)
extract_noise()
