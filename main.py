import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

color = cv2.imread(sys.argv[1])
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

def add(dest, src):
    (x, y) = dest.shape
    for i in range(x):
        for j in range(y):
            if src[i][j] != 0:
                dest[i][j] = src[i][j]

# 63, 127, 191

_, thres1 = cv2.threshold(img,  31,  31, cv2.THRESH_BINARY)
_, thres2 = cv2.threshold(img,  63,  63, cv2.THRESH_BINARY)
_, thres3 = cv2.threshold(img, 127, 127, cv2.THRESH_BINARY)

result = np.zeros(img.shape, dtype=np.uint8)

add(result, thres1)
add(result, thres2)
add(result, thres3)

_, thres1 = cv2.threshold(img,  63,  64, cv2.THRESH_BINARY)
_, thres2 = cv2.threshold(img, 127, 127, cv2.THRESH_BINARY)
_, thres3 = cv2.threshold(img, 191, 191, cv2.THRESH_BINARY)

result2 = np.zeros(img.shape, dtype=np.uint8)

add(result2, thres1)
add(result2, thres2)
add(result2, thres3)

plt.subplot(2, 2, 1), plt.imshow(color)
plt.subplot(2, 2, 2), plt.imshow(img, cmap='gray')
plt.subplot(2, 2, 3), plt.imshow(result2, cmap='gray')
plt.subplot(2, 2, 4), plt.imshow(result, cmap='gray')

plt.show()
