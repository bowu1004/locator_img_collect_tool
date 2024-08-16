from PIL import Image
import cv2
import numpy as np

def calculate_mean_hash(image_path, size=480):
    # 读取图像并调整大小
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算像素平均值
    avg_pixel = np.mean(gray)

    # 生成哈希值
    hash_value = ""
    for i in range(size):
        for j in range(size):
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

if __name__ == '__main__':

    # 使用示例
    # image_path1 = "dataset/train/coco_cola/00000.png"
    # image_path2 = "dataset/train/coco_cola/00009.png"
    # hash1 = calculate_mean_hash(image_path1)
    # hash2 = calculate_mean_hash(image_path2)
    # distance = hamming_distance(hash1, hash2)
    # print("汉明距离：", distance)

    import os
    coco_cola_root_dir = r'dataset/train/coco_cola'
    mo_zhua_root_dir = r'dataset/train/mo_zhua'

    #计算不同类别之间的汉明距离
    hanming_distance = []
    for i in os.listdir(coco_cola_root_dir):
        curr_coco_cola_path = os.path.join(coco_cola_root_dir, i)
        for j in os.listdir(mo_zhua_root_dir):
            curr_mo_zhua_path = os.path.join(mo_zhua_root_dir, j)
            hash1 = calculate_mean_hash(curr_coco_cola_path)
            hash2 = calculate_mean_hash(curr_mo_zhua_path)
            distance = hamming_distance(hash1, hash2)
            hanming_distance.append(distance)
    print(max(hanming_distance))
    print(min(hanming_distance))


    #计算可乐同类别之间的汉明距离
    hanming_distance = []
    file_names = os.listdir(coco_cola_root_dir)
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            # 比较 file_names[i] 和 file_names[j]
            image1_path = os.path.join(coco_cola_root_dir, file_names[i])
            image2_path = os.path.join(coco_cola_root_dir, file_names[j])
            hash1 = calculate_mean_hash(image1_path)
            hash2 = calculate_mean_hash(image2_path)
            distance = hamming_distance(hash1, hash2)
            hanming_distance.append(distance)
    print(max(hanming_distance))
    print(min(hanming_distance))


    #计算魔抓同类别之间的汉明距离
    hanming_distance = []
    file_names = os.listdir(mo_zhua_root_dir)
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            # 比较 file_names[i] 和 file_names[j]
            image1_path = os.path.join(mo_zhua_root_dir, file_names[i])
            image2_path = os.path.join(mo_zhua_root_dir, file_names[j])
            hash1 = calculate_mean_hash(image1_path)
            hash2 = calculate_mean_hash(image2_path)
            distance = hamming_distance(hash1, hash2)
            hanming_distance.append(distance)
    print(max(hanming_distance))
    print(min(hanming_distance))