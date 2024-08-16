'''采集指定的object的模板'''
from ultralytics import YOLO
import os
import cv2
import pyrealsense2 as rs
from depth_restore.solver import *
import keyboard
from camera import realsense
from utils import *
def run_yolov8_inference():
    # 实例化定位器模型
    model = YOLO(r"CD_pth/object.pt")
    solver = Solver()
    solver.init_weights(r"CD_pth/CDNet.pth")
    # 实例化相机类
    camera = realsense.RealSenseCamera()
    # 启动相机流并获取内参信息
    camera.start_camera()
    while True:
        # 相机读取RGB+D流
        color_img, depth_img, _, point_cloud, depth_frame = camera.read_align_frame()
        # 进行深度学重建
        depth_img_restore = cv2.cvtColor(solver.test(img=color_img), cv2.COLOR_GRAY2RGB)

        if len(color_img):
            # TODO: 将realsense整合到yolov8，预计能加快1ms的速度并节约内存消耗
            results = model(color_img, depth_img_restore, conf=0.25)
            '''获取bbox的坐标，并抠出bbx'''
            ori_list = results[0].boxes.xyxy.cpu().numpy()
            if ori_list.shape[0] == 1:
                x1, y1, x2, y2 = int(ori_list[0][0]), int(ori_list[0][1]), int(ori_list[0][2]), int(ori_list[0][3])
                roi_img = color_img[y1: y2, x1: x2]
                cv2.imshow('roi', roi_img)


            annotated_frame = results[0].plot()

            cv2.imshow('results', annotated_frame)

            cv2.waitKey(1)

            if keyboard.is_pressed('F3'):
                cv2.imwrite('dataset/desk_arm/mozhua/mozhua.jpg', roi_img)
                print('保存成功')


            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break



if __name__ == "__main__":
    run_yolov8_inference()
