import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0],
                [0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0],
                [0, 255, 255, 255, 255, 255, 0, 0, 255, 255, 0],
                [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 0],
                [0, 0, 0, 0, 255, 0, 0, 255, 255, 255, 0],
                [0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0],
                [0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ])
img = np.uint8(img)

kernel = np.array ([[0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 0]])
kernel = np.uint8(kernel)

hitMissKernel = np.array ([ [0, 1, 0],
                            [0, 1, 1],
                            [0, 0, 0]])
hitMissKernel = np.uint8(kernel)

def showResult(title, data):
    print(title)
    print(data)
    print('\n')

if __name__ == "__main__":

    erosion = cv2.erode(img,kernel, iterations = 1)
    dilation = cv2.dilate(img,kernel, iterations = 1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    hitOrMiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, hitMissKernel)

    showResult('Citra Awal', img)
    showResult('Kernel', kernel)
    showResult('Erosi', erosion)
    showResult('Dilasi', dilation)
    showResult('Opening', opening)
    showResult('Closing', closing)
    showResult('Hit or Miss', hitOrMiss)