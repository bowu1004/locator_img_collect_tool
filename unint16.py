import cv2
import numpy as np
img_path = r'dataset/train/coco_cola/00000.png'
img_uint8 = cv2.imread(img_path)
img_uint16 = img_uint8.astype(np.uint16)
img_float64 = img_uint8.astype(np.float64)
cv2.imshow('uint8', img_uint8)
cv2.imshow('uint16', img_uint16 * 50)
cv2.imshow('float64', img_float64)
cv2.waitKey(0)
cv2.destroyAllWindows()