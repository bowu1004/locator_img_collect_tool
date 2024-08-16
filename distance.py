import cv2
import numpy as np

def img2vector(img_gray):
    # 获取非0像素的坐标和值
    non_zero_pixels = np.transpose(np.nonzero(img_gray))
    non_zero_pixel_values = img_gray[non_zero_pixels[:, 0], non_zero_pixels[:, 1]]

    # 输出非0像素的数量和均值
    print("非0像素的数量：", len(non_zero_pixel_values))

    # 归一化非0像素的值
    normalized_values = np.linalg.norm(non_zero_pixel_values)
    normalized_vector = non_zero_pixel_values / normalized_values

    return normalized_vector

def padding_0(vector1, vector2):
    len1 = len(vector1)
    len2 = len(vector2)
    # 将较短的向量用零填充到与较长的向量相同的长度
    if len1 < len2:
        vector1 = np.concatenate([vector1, np.zeros(len2 - len1)])
    elif len2 < len1:
        vector2 = np.concatenate([vector2, np.zeros(len1 - len2)])

    return vector1, vector2

if __name__ == '__main__':

    coco_cola = cv2.imread('dataset/train/coco_cola/00000.png')
    coco_cola_gray = cv2.cvtColor(coco_cola, cv2.COLOR_BGR2GRAY)
    mozhua = cv2.imread('dataset/train/coco_cola/00001.png')
    mozhua_gray = cv2.cvtColor(mozhua, cv2.COLOR_BGR2GRAY)


    coco_cola_vector = img2vector(coco_cola_gray)
    mozhua_vector = img2vector(mozhua_gray)
    #进行维度调整
    coco_cola_vector, mozhua_vector = padding_0(coco_cola_vector, mozhua_vector)
    # 计算欧氏距离
    euclidean_distance = np.linalg.norm(coco_cola_vector - mozhua_vector)
    # 输出欧氏距离
    print("欧氏距离：", euclidean_distance)


    # 计算余弦相似度
    cos_sim = np.dot(coco_cola_vector, mozhua_vector) / (np.linalg.norm(coco_cola_vector) * np.linalg.norm(mozhua_vector))
    # 输出余弦相似度
    print("余弦相似度：", cos_sim)