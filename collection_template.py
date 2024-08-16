'''采集指定的object的模板'''
from ultralytics import YOLO
import os
import cv2
import pyrealsense2 as rs
from solver import *
import keyboard
def run_yolov8_inference():
    root_dir = r'dataset/huojia'
    template_feature = {}
    names = {}
    for index, i in enumerate(os.listdir(root_dir)):
        curr_class_path = os.path.join(root_dir, i)
        curr_template_img_path = os.path.join(curr_class_path, os.listdir(curr_class_path)[0])
        template_feature[i] = cv2.imread(curr_template_img_path)
        names[index] = i
    # Load a model
    model = YOLO(r"CD_pth/object.pt")
    solver = Solver()
    pipeline = rs.pipeline()
    align_to = rs.stream.color
    align = rs.align(align_to)
    config = rs.config()
    D400_imgWidth, D400_imgHeight = 640, 480  # Best depth performance resolution is (848,480)
    config.enable_stream(rs.stream.color, D400_imgWidth, D400_imgHeight, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, D400_imgWidth, D400_imgHeight, rs.format.z16, 30)

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
        frame_data = np.asanyarray(aligned_color_frame.get_data())
        deep_data = np.asanyarray(aligned_depth_frame.get_data())
        # deep_data2 = cv2.cvtColor(deep_data, cv2.COLOR_GRAY2BGR)

        deep_data_depth_esi = solver.test(ckpt_path=r'CD_pth/CDNet.pth', batch_size=1, test_thread_num=8, img=frame_data)
        deep_data3 = cv2.cvtColor(deep_data_depth_esi, cv2.COLOR_GRAY2BGR)

        if len(frame_data):
            # TODO: 将realsense整合到yolov8，预计能加快1ms的速度并节约内存消耗
            results = model(frame_data, deep_data3, conf=0.25)
            '''获取bbox的坐标，并抠出bbx'''
            ori_list = results[0].boxes.xyxy.cpu().numpy()
            if ori_list.shape[0] == 1:
                x1, y1, x2, y2 = int(ori_list[0][0]), int(ori_list[0][1]), int(ori_list[0][2]), int(ori_list[0][3])
                roi_img = frame_data[y1: y2, x1: x2]
                cv2.imshow('roi', roi_img)


            annotated_frame = results[0].plot()

            cv2.imshow('results', annotated_frame)
            cv2.waitKey(1)

            if keyboard.is_pressed('F3'):
                cv2.imwrite('dataset/desk_arm/mozhua/mozhua.jpg', roi_img)
                print('保存成功')


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    #     else:
    #         break
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    run_yolov8_inference()
