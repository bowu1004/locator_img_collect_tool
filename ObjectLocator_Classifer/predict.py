import os
import cv2
import keyboard
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
from depth_restore.solver import *
import torch.nn as nn
from camera import realsense
from utils import *
import time

'''这个代码已经做了封装'''
'''这是用巴氏距离、线性相似和卡方分布计算整个RGB bbx作分类器'''
'''将分类器和定位器集成，分类器的结果更新到定位器中'''

def run_yolov8_inference():
    root_dir = r'dataset/desk_arm'
    #获取模板以及类别名
    template_feature, names = get_template(root_dir)
    #实例化定位器模型
    model = YOLO(r"CD_pth/object.pt")
    solver = Solver()
    solver.init_weights(r"CD_pth/CDNet.pth")
    # 实例化相机类
    camera = realsense.RealSenseCamera()
    # 启动相机流并获取内参信息
    camera.start_camera()

    while True:
        #相机读取RGB+D流
        color_img, depth_img, _, point_cloud, depth_frame = camera.read_align_frame()
        #进行深度学重建
        depth_img_restore = cv2.cvtColor(solver.test(img=color_img), cv2.COLOR_GRAY2RGB)
        if len(color_img):
            # #进行定位器推理
            start_time = time.time()
            results = model(color_img, depth_img_restore, conf=0.55)
            end_time = time.time()
            print("定位器的推理时间为:{}s".format(end_time - start_time))
            if results[0].masks is None:
                continue
            #获取bbox的坐标，截取出object的roi
            ori_list = results[0].boxes.data.tolist()
            # 进行分类器推理
            start_time2 = time.time()
            #遍历当前帧的多个bbox的分类结果
            total_cls_results = {}
            for bbox in ori_list:

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cr_img = color_img[y1:y2, x1:x2]

                #将比较的图片统一至相同的大小
                template_feature["yolo_predict"] = cr_img
                img_update_dict = resize(template_feature)
                #建立直方图#
                output_hist_dict = create_rgb_hist(img_update_dict)
                #根据RGB直方图计算巴士距离、相似性以及卡方分布
                bashi_dict, similarity_dict, kafang_dict = hist_compare(output_hist_dict)
                #输出分类结果
                cls_results = compare_output(bashi_dict, similarity_dict, kafang_dict)

                total_cls_results[cls_results] = bbox
            end_time2 = time.time()
            print("分类器器的推理时间为:{}s".format(end_time2 - start_time2))

            if not all(key == None for key in total_cls_results.keys()):
                #将当前模板的类别更新覆盖Yolo训练集的类别
                results[0].names = names
                #根据分类器的结果获取对应的索引，进行更新定位器推理结果中的类别
                annotated_frame = results[0].plot2(class_results=total_cls_results)

                cv2.imshow('results', annotated_frame)
                cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    run_yolov8_inference()
