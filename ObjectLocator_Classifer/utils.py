import cv2
import numpy as np
import os

def color_correction(color_image):
    """
    D435相机因python版本原因出现色差，进行色差矫正
    :param color_image:相机读出来的RGB
    :return:进行色差矫正的RGB
    """
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 8) % 180  # %180:

    color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return color_image

def get_template(root_dir):
    '''
    :param root_dir(模板的根路径):
    :return
    template_feature: 读取的模板
    names: 模板的类别名(后面更新定位器的类别)
    '''
    template_feature = {}
    names = {}
    for index, i in enumerate(os.listdir(root_dir)):
        curr_class_path = os.path.join(root_dir, i)
        curr_template_img_path = os.path.join(curr_class_path, os.listdir(curr_class_path)[0])
        template_feature[i] = cv2.imread(curr_template_img_path)
        names[index] = i
    return template_feature, names


def resize(template_feature_dict):
    """
    :param template_feature_dict: 模板以及当前帧的多个bbox的roi
    :return: 进行尺寸调整后的模板以及bbox的roi
    """
    min_width = 1000
    min_height = 1000
    # 比较宽度和高度，确定调整后的尺寸
    for img_dict_key in template_feature_dict:
        img = template_feature_dict[img_dict_key]
        curr_height, curr_width = img.shape[0], img.shape[1]
        if curr_height < min_height:
            min_height = curr_height
        if curr_width < min_width:
            min_width = curr_width

    # 调整图片的尺寸
    img_update_dict = {key: cv2.resize(template_feature_dict[key], (min_height, min_width))for key in template_feature_dict}

    return img_update_dict


def MSE(img1, img2):
    """

    :param img1:
    :param img2:
    :return: 两个img之间的均方误差
    """
    mse = np.mean( (img1 - img2) ** 2 )
    return mse


def create_rgb_hist(template_feature):
    """
    创建直方图
    :param template_feature:模板以及当前帧的多个bbox的roi
    :return:
    """
    output_dict = {}
    for image_dict_key in template_feature:
        image = template_feature[image_dict_key]
        '''先进行HSV做亮度V的均衡'''
        image = handle_img(image)
        """"创建 RGB 三通道直方图（直方图矩阵）"""
        h, w, c = image.shape
        # 创建一个（16*16*16,1）的初始矩阵，作为直方图矩阵
        # 16*16*16的意思为三通道每通道有16个bins
        rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)
        bsize = 256 / 16
        for row in range(h):
            for col in range(w):
                b = image[row, col, 0]
                g = image[row, col, 1]
                r = image[row, col, 2]
                # 人为构建直方图矩阵的索引，该索引是通过每一个像素点的三通道值进行构建
                index = int(b / bsize) * 16 * 16 + int(g / bsize) * 16 + int(r / bsize)
                # 该处形成的矩阵即为直方图矩阵
                rgbhist[int(index), 0] += 1

        output_dict[image_dict_key] = rgbhist
        # plt.ylim([0, 10000])
        # plt.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.3)
    return output_dict


def hist_compare(output_hist_dict):
    """直方图比较函数"""
    '''# 创建第一幅图的rgb三通道直方图（直方图矩阵）
    hist1 = create_rgb_hist(image1)
    # 创建第二幅图的rgb三通道直方图（直方图矩阵）
    hist2 = create_rgb_hist(image2)'''
    bashi_dict = {}
    similarity_dict = {}
    kaafang_dict = {}
    yolo_predict_hist = output_hist_dict["yolo_predict"]
    for img_dict_key in output_hist_dict:
        if img_dict_key != "yolo_predict":
            image = output_hist_dict[img_dict_key]
            match1 = cv2.compareHist(yolo_predict_hist, image, cv2.HISTCMP_BHATTACHARYYA)
            match2 = cv2.compareHist(yolo_predict_hist, image, cv2.HISTCMP_CORREL)
            match3 = cv2.compareHist(yolo_predict_hist, image, cv2.HISTCMP_CHISQR)
            bashi_dict[img_dict_key] = match1
            similarity_dict[img_dict_key] = match2
            kaafang_dict[img_dict_key] = match3

    return bashi_dict, similarity_dict, kaafang_dict

def handle_img(img):
    """
    对HSV颜色空间中的V（亮度做一下均衡)
    :param img:均衡后的图像，再转回BGR
    :return:
    """
    # img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #对HSV颜色空间中的V（亮度做一下均衡），再转会BGR
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def compare_output(bashi_dict, similarity_dict, kafang_dict):
    """

    :param bashi_dict:当前帧的多个roi与模板之间的巴士距离的字典数据
    :param similarity_dict:前帧的多个roi与模板之间的线性相似性的字典数据
    :param kafang_dict:前帧的多个roi与模板之间的卡方分布的字典数据
    :return:符合三种约束条件的模板名称
    """
    min_bashi_key = min(bashi_dict, key=bashi_dict.get)
    max_similarity_key = max(similarity_dict, key=similarity_dict.get)
    min_kafang_key = min(kafang_dict, key=kafang_dict.get)

    if min_bashi_key == max_similarity_key == min_kafang_key:
        return min_bashi_key