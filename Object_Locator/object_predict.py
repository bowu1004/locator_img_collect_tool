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
    #实例化定位器模型
    model = YOLO(r"CD_pth/coco_pre_best.pt")
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
            results = model(color_img, depth_img_restore, conf=0.25)
            end_time = time.time()
            print("定位器的推理时间为:{}s".format(end_time - start_time))
            if results[0].masks is None:
                continue
            annotated_frame = results[0].plot()

            cv2.imshow('results', annotated_frame)
            cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    run_yolov8_inference()
