#!/usr/bin/env python3

import cv2
import numpy as np

sign_cascade = cv2.CascadeClassifier("classifier/cascade.xml")

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

signs = sign_cascade.detectMultiScale(gray, 1.1, 5)
print(signs)
for sign in range(np.size(signs, 0)):
    x_pos, y_pos, width, height = signs[sign]
    cv2.imwrite(str(sign) + ".png", img[sign[0]:sign[0] + sign[2] + sign[1], sign[1] + sign[3]])


cv2.destroyAllWindows()
