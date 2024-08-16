import cv2
import numpy as np
def img2_minAreaRect_roi(img):
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


def imgpath2_minAreaRect_roi(img_path):
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


def resize(template_feature_dict):
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


def upsample(template_feature_dict):
    max_width = 0
    max_height = 0
    # 比较宽度和高度，确定调整后的尺寸
    for img_dict_key in template_feature_dict:
        img = template_feature_dict[img_dict_key]
        curr_height, curr_width = img.shape[0], img.shape[1]
        if curr_height > max_height:
            max_height = curr_height
        if curr_width > max_width:
            max_width = curr_width

    # 调整图片的尺寸
    img_last_list = {key: cv2.resize(template_feature_dict[key], (max_height, max_width), interpolation=cv2.INTER_LINEAR) for key in template_feature_dict}

    return img_last_list

def padding(img_list):
    padding_list = []
    target_height = 0
    target_width = 0
    '''遍历获取所有图像中的最大的宽、高值'''
    for img in img_list:
        img_height, img_width = img.shape[:2]
        if img_height > target_height:
            target_height = img_height
        if img_width > target_width:
            target_width = img_width
    '''所有图片统一padding至相同的size'''
    for img in img_list:
        img_height, img_width = img.shape[:2]
        if not ((img_height == target_height) and (img_width == target_width)):
            # 计算需要添加的填充
            ratio = min(target_width / img_width, target_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            # 调整图像尺寸并添加填充
            resized_image = cv2.resize(img, (new_width, new_height))
            top = (target_height - new_height) // 2
            bottom = target_height - new_height - top
            left = (target_width - new_width) // 2
            right = target_width - new_width - left
            padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padding_list.append(padded_image)
        else:
            padding_list.append(img)
    return padding_list

def MSE(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse


def create_rgb_hist(template_feature):
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
    # 进行三种方式的直方图比较
    # match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    # match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    # print("巴氏距离：%s, 相关性：%s, 卡方：%s" % (match1, match2, match3))
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
    # img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #对HSV颜色空间中的V（亮度做一下均衡），再转会BGR
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def compare2_output(bashi_dict, similarity_dict, kafang_dict):
    min_bashi_key = min(bashi_dict, key=bashi_dict.get)
    max_similarity_key = max(similarity_dict, key=similarity_dict.get)
    min_kafang_key = min(kafang_dict, key=kafang_dict.get)

    if min_bashi_key == max_similarity_key == min_kafang_key:
        return min_bashi_key