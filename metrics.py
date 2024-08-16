import os
import numpy as np
import cv2
'''求解最小外接矩形'''
def minAreaRect_roi(img_path):
    # 读入图像
    img = cv2.imread(img_path)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值方法进行二值化处理
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # 找到轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 绘制最小外接矩形
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # 创建黑色背景图像
    background = np.zeros_like(img)

    # 在背景图像上绘制最小外接矩形
    cv2.drawContours(background, [box], 0, (255, 255, 255), -1)

    # 提取感兴趣的区域
    roi = cv2.getRectSubPix(img, (int(rect[1][0]), int(rect[1][1])), rect[0])
    return roi


'''归一化'''
def standardization(image):
    mean = np.mean(image)
    std = np.std(image)
    standardized_image = (image - mean) / std
    return standardized_image

'''将两个图片统一至相同的size'''
def resize(img1, img2):
    # 转灰度
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 获取图片的宽度和高度
    height_img1, width_img1 = img1.shape
    height_img2, width_img2 = img2.shape

    # 比较宽度和高度，确定调整后的尺寸
    new_width = min(width_img1, width_img2)
    new_height = min(height_img1, height_img2)

    # 调整图片的尺寸
    resized_img1 = cv2.resize(img1, (new_width, new_height))
    resized_img2 = cv2.resize(img2, (new_width, new_height))

    '''归一化'''
    resized_img1 = standardization(resized_img1)
    resized_img2 = standardization(resized_img2)

    return resized_img1, resized_img2


def MSE(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse

'''计算可乐和魔抓之间的相似度'''
coco_cola_dir = r'dataset/train/coco_cola'
mozhua_dir = r'dataset/train/mo_zhua'
mse_list = []
for i in os.listdir(coco_cola_dir):
    curr_coco_cola_path = os.path.join(coco_cola_dir, i)
    for j in os.listdir(mozhua_dir):
        curr_mozhua_path = os.path.join(mozhua_dir, j)
        '''开始比较'''
        coco_cola = minAreaRect_roi(curr_coco_cola_path)
        mozhua = minAreaRect_roi(curr_mozhua_path)
        coco_cola, mozhua = resize(coco_cola, mozhua)
        mse = MSE(coco_cola, mozhua)
        mse_list.append(mse)
print(mse_list)

'''计算可乐和可乐之间的相似度'''
coco_cola_dir = r'dataset/train/coco_cola'
mozhua_dir = r'dataset/train/mo_zhua'
mse_list = []
for i in os.listdir(coco_cola_dir):
    curr_coco_cola_path = os.path.join(coco_cola_dir, i)
    for j in os.listdir(coco_cola_dir)[1:]:
        next_coco_cola_path = os.path.join(coco_cola_dir, j)
        '''开始比较'''
        coco_cola = minAreaRect_roi(curr_coco_cola_path)
        next_coco_cola = minAreaRect_roi(next_coco_cola_path)
        coco_cola, next_coco_cola = resize(coco_cola, next_coco_cola)
        mse = MSE(coco_cola, next_coco_cola)
        # if mse < 1.0:
        #     print("当前比较的图片是：{}和{}，mse是：{}".format(curr_coco_cola_path, next_coco_cola_path, mse))

        mse_list.append(mse)
print(mse_list)
print(len(mse_list))

'''计算魔抓和魔抓之间的相似度'''
mozhua_dir = r'dataset/train/mo_zhua'
mse_list = []
for i in os.listdir(mozhua_dir):
    curr_mozhua_path = os.path.join(mozhua_dir, i)
    for j in os.listdir(mozhua_dir)[1:]:
        next_mozhua_path = os.path.join(mozhua_dir, j)
        '''开始比较'''
        mozhua = minAreaRect_roi(curr_mozhua_path)
        next_mozhua = minAreaRect_roi(next_mozhua_path)
        mozhua, next_mozhua = resize(mozhua, next_mozhua)
        mse = MSE(mozhua, next_mozhua)
        # if mse < 1.0:
        #     print("当前比较的图片是：{}和{}，mse是：{}".format(curr_coco_cola_path, next_coco_cola_path, mse))

        mse_list.append(mse)
print(mse_list)
print(len(mse_list))