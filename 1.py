import numpy as np
import matplotlib.pyplot as plt
import cv2
import urllib.request as urllib
import imutils
from lib import *


block_size = 5
C = 1

img = cv2.imread('img.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = np.zeros_like(img)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,61,3)

contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

box = cv2.minAreaRect(contours[1])
box = cv2.boxPoints(box)

approx = cv2.approxPolyDP(contours[1], 0.01*cv2.arcLength(contours[1], True), True)


corner = find_corner_by_rotated_rect(box,approx)
image = four_point_transform(img,corner)
wrap = four_point_transform(thresh,corner)
res = np.zeros_like(wrap, dtype=np.uint8)
cv2.imshow("Wrap",wrap)

contours = cv2.findContours(wrap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
count = 0
tickcontours = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if 0.8 <= w/float(h) <= 1.2 and w > 30 and h > 30:
        # cv2.drawContours(img, [cnt], 0, (0,255,0), 2)
        tickcontours.append(cnt)

ticked = {}     

tickcontours = sort_contours(tickcontours, method='top-to-bottom')[0]
for q, i in enumerate(np.arange(0, len(tickcontours), 5)):
    cnts = sort_contours(tickcontours[i:i+5])[0]
    max_points = 0
    pos = 0    
    for j in range(len(cnts)):
        mask = np.zeros_like(wrap, dtype=np.uint8)
        cv2.drawContours(mask, [cnts[j]], -1, 255, -1)
        mask = cv2.bitwise_and(wrap, wrap, mask=mask)
        count = cv2.countNonZero(mask)
        print(count)
        if count > max_points:
            max_points = count
            pos = j

    ticked[q] = pos
    

print(ticked)
        

# cv2.imshow('mask', res)
# cv2.imshow('img', img)
cv2.waitKey(0)

cv2.destroyAllWindows()