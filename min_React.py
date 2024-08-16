import cv2
import numpy as np
import os
'''求解最小外接矩形'''
def minAreaRect_roi(img_path):
    # 读入图像
    # img = img_path
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

def minAreaRect_roi2(img_path):
    # 读入图像
    img = img_path
    # img = cv2.imread(img_path)

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

    return resized_img1, resized_img2

def img2vector(img):
    img = img.flatten()  # 对图像进行降维操作，方便算法计算
    img = img / 255.
    return img

def get_shortest_dimensions(image_list):
    shortest_width = float('inf')
    shortest_height = float('inf')

    for image in image_list:
        height, width = image.shape[0], image.shape[1]
        if width < shortest_width:
            shortest_width = width
        if height < shortest_height:
            shortest_height = height

    return shortest_width, shortest_height

if __name__ == '__main__':
    def calculate_mean_hash_img(gray, size=480):
        # 读取图像并调整大小
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (size, size))

        # 转换为灰度图像
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        w, h = gray.shape[0], gray.shape[1]
        # 计算像素平均值
        avg_pixel = np.mean(gray)

        # 生成哈希值
        hash_value = ""
        for i in range(w):
            for j in range(h):
                if gray[i, j] < avg_pixel:
                    hash_value += "0"
                else:
                    hash_value += "1"

        return hash_value


    def hamming_distance(hash1, hash2):
        # 计算汉明距离
        distance = 0
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                distance += 1
        return distance

    roi = minAreaRect_roi(r'dataset/train/coco_cola/00000.png')
    roi2 = minAreaRect_roi(r'dataset/train/coco_cola/00002.png')

    resized_img1, resized_img2 = resize(roi, roi2)
    hash1 = calculate_mean_hash_img(resized_img1)
    hash2 = calculate_mean_hash_img(resized_img2)
    distance = hamming_distance(hash1, hash2)
    print(distance)
    cv2.imshow('a', resized_img1)
    cv2.imshow('b', resized_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()