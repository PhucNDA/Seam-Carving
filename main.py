import numpy as np
import math
import time
import os
import cv2
from scipy import ndimage as ndi 
from numba import jit 
import SeamCarving as SC
from CreateMask import getMaskObject

def main():
    img = cv2.imread(".\img-original\img-original-tower.png")
    img1, img2 = img.copy(), img.copy()
    mask1 = getMaskObject(img1)
    print("Create Mask 1, DONE!!!")
    mask2 = getMaskObject(img2)
    print("Create Mask 2, DONE!!!")
    h, w, c=img.shape
    start = time.time()
    img=SC.removeObjectfromMask(img, objectMaskProtect = mask2, objectMaskDelete = mask1)
    numberResult = math.ceil(len(os.listdir('./img-result')) / 2) + 1
    cv2.imwrite('.\img-result\img-result0' + str(numberResult) + '.png', img)
    revertSize((h,w,c), img, numberResult)
    end = time.time()
    print(f"Time consuming in order to remove object use Seam Carving: {end - start}s")

def revertSize(originalShape, curImage, numberResult):
    h,w,c=originalShape
    print(curImage.shape)
    hs,ws,cs=curImage.shape
    curImage=SC.enlargeImage(curImage,w-ws)
    cv2.imwrite('.\img-result\img-final' + str(numberResult) + '.png', curImage)

if __name__ == "__main__":
    main()