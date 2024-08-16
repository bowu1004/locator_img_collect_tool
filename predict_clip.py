import argparse
import os

import cv2
import keyboard
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
from solver import *
import pickle
from torchvision.models import resnet18, resnet34
import json
import faiss
import torch.nn as nn
from utils import *
'''这个代码已经做了封装'''
'''这是用巴氏距离和线性相似度计算整个RGB bbx作分类器'''
'''将分类器和定位器集成，分类器的结果更新到定位器中'''
'''To update:当前帧出现大于2的object，选取位于最中间的两个object'''
import torch
import os
import random
import string
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor

import os
import datetime


def save_file_to_timed_folder(out_path, filename, frame_data):
    # 获取当前时间，并格式化为年月日时分秒的形式，作为文件夹名
    folder_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建文件夹路径
    path = os.path.join(out_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    cv2.imwrite(file_path, frame_data)
    return path



# 如果类别字典中没有该类别，则在字典中添加该类别
def generate_random_filename(extension='png'):
    # 生成一个由10个随机字母和数字组成的字符串
    chars = string.ascii_letters + string.digits
    filename = ''.join(random.choice(chars) for _ in range(10))
    return f"{filename}.{extension}"
def add_category_if_not_in_dict(category_dict, category_name):
    if category_name not in category_dict.values():
        new_key =max(category_dict.keys())+1 if category_dict.keys() else 1
        category_dict[new_key]=category_name
    else:
        pass

## 传入图片搜索5000种类图像向量库
def image_search(image, clip_model,clip_processor,id2filename,index,k=1):

    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    image_features = image_features.detach().numpy()
    D, I = index.search(image_features, k)  # 实际的查询

    cls_name = [[id2filename[str(j)] for j in i] for i in I]

    return  D, cls_name

def run_yolov8_inference(faiss_path,out_path):
    base_path = os.getcwd()
    # 设置路径
    yolo_model_path = os.path.join(base_path, "CD_pth", "object.pt")
    clip_model_path = os.path.join(base_path, "model_path", "clip_model")
    faiss_category = os.path.join(faiss_path, "category_name.json")
    faiss_vector = os.path.join(faiss_path, "image_faiss.index")

    # 加载yolo模型文件，加载clip模型文件
    yolo_model = YOLO(yolo_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    # 加载商品向量对应的类别名称
    with open(faiss_category, 'r') as json_file:
        id2filename = json.load(json_file)
    # 加载商品图片向量
    index = faiss.read_index(faiss_vector)

    solver = Solver()
    pipeline = rs.pipeline()
    align_to = rs.stream.color
    align = rs.align(align_to)
    config = rs.config()
    D400_imgWidth, D400_imgHeight = 1920, 1080
    target_imgWidth, target_imgHeight = 848, 480
    width_scale = D400_imgWidth/target_imgWidth
    height_scale =  D400_imgHeight/target_imgHeight
    # D400_imgWidth, D400_imgHeight = 848, 480
    # D400_imgWidth, D400_imgHeight = 1280, 720  # Best depth performance resolution is (848,480)
    # D400_imgWidth, D400_imgHeight = 640, 480
    config.enable_stream(rs.stream.color, D400_imgWidth, D400_imgHeight, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, D400_imgWidth, D400_imgHeight, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    input = 'object'
    while True:
        for _ in range(2):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_intrin = (
            aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        )
        color_intrin = (
            aligned_color_frame.profile.as_video_stream_profile().intrinsics
        )

        profile = aligned_frames.get_profile()



        if not aligned_depth_frame or not aligned_color_frame:
            raise Exception("[info] No D435 data.")
        # Image numpy array
        origin_frame_data = np.asanyarray(aligned_color_frame.get_data())
        # frame_data = np.asanyarray(aligned_color_frame.get_data())
        # 保存图像到文件
        # filename = generate_random_filename()
        #
        # file_path = os.path.join(out_path, filename)
        # cv2.imwrite(file_path, origin_frame_data)
        # resize
        frame_data = cv2.resize(origin_frame_data, (target_imgWidth, target_imgHeight))
        # frame_data = origin_frame_data.resize(848, 480)


        deep_data_depth_esi = solver.test(ckpt_path=r'CD_pth/CDNet.pth', batch_size=1, test_thread_num=8, img=frame_data)
        deep_data3 = cv2.cvtColor(deep_data_depth_esi, cv2.COLOR_GRAY2BGR)

        if len(frame_data):
            # TODO: 将realsense整合到yolov8，预计能加快1ms的速度并节约内存消耗
            results = yolo_model(frame_data, deep_data3, conf=0.55)

            if results[0].masks is None:
                continue
            '''获取bbox的中心点x坐标，并抠出最中间的两个bbox'''
            ori_x_list = results[0].boxes.xywh[:, 0]
            # 找到张量中值的排序索引
            ori_list = results[0].boxes.data.tolist()
            '''将当前帧的多个bbox的分类结果进行统计'''
            total_cls_results = []
            # total_cls_results = {}
            category_dict={}
            # 保存图像到文件
            # filename = generate_random_filename()
            # file_path=os.path.join(out_path,filename)
            # cv2.imwrite(file_path, frame_data)
            # 保存大图到文件以及小图类别文件夹
            filename = generate_random_filename()
            # out_path_time=save_file_to_timed_folder(out_path, filename, frame_data)
            file_path = os.path.join(out_path, filename)
            cv2.imwrite(file_path, origin_frame_data)

            for bbox in ori_list:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cr_img = frame_data[y1:y2, x1:x2]
                # 此处的逻辑是：将resize后的小框位置还原到大图上，然后抠出小框进行分类
                new_x1 = int(x1 * width_scale)
                new_y1 = int(y1 * height_scale)
                new_x2 = int(x2 * width_scale)
                new_y2 = int(y2 * height_scale)
                cr_img_origin = origin_frame_data[new_y1:new_y2, new_x1:new_x2]

                # 保存小图到文件
                # filename = generate_random_filename()
                # file_path = os.path.join(out_path, filename)
                # cv2.imwrite(file_path, cr_img_origin)

                image_rgb=cv2.cvtColor(cr_img_origin, cv2.COLOR_BGR2RGB)
                image=Image.fromarray(image_rgb)
                D,cls_results_dict=image_search(image, clip_model,clip_processor,id2filename,index,k=4)
                cls_result=cls_results_dict[0][0]
                # 保存小图到文件,并按照类别名称
                filename_xiaotu = generate_random_filename()
                filepath_xiaotu=os.path.join(out_path, cls_result)
                if not os.path.exists(filepath_xiaotu):
                    os.makedirs(filepath_xiaotu)
                file_path_xiaotu = os.path.join(filepath_xiaotu, filename_xiaotu)
                cv2.imwrite(file_path_xiaotu, cr_img_origin)

                add_category_if_not_in_dict(category_dict,cls_result)
                total_cls_results.append({cls_result: bbox})

            if len(total_cls_results) >=  1:
                '''将当前模板的类别更新覆盖Yolo训练集的类别'''
                results[0].names = category_dict
                # results[0].boxes.cls[:]=0
                # results[0].boxes.conf[:]=1.0
                '''根据分类器的结果获取对应的索引，进行更新覆盖Yolo推理结果中的分类cls'''
                # total_class_index = [list(names.values()).index(i) for i in total_cls_results]
                annotated_frame = results[0].plot3(class_results=total_cls_results)
                # filepath_keshihua = os.path.join(out_path, 'keshihua')
                # if not os.path.exists(filepath_keshihua):
                #     os.makedirs(filepath_keshihua)
                # filepath_keshihua_pic=os.path.join(filepath_keshihua, filename)
                # cv2.imwrite(filepath_keshihua_pic, annotated_frame)
                cv2.imshow('results', annotated_frame)

                cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create Clip Vector Library", add_help=True)
    parser.add_argument("--faiss_path", type=str, default=r"./output/clip_faiss", help="faiss vector library path")
    parser.add_argument("--out_path", type=str, default=r"./output/data", help="output path")
    args = parser.parse_args()
    # base_path = os.getcwd()
    # out_path = os.path.join(base_path, "output","data")
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    run_yolov8_inference(faiss_path=args.faiss_path,out_path=args.out_path)
