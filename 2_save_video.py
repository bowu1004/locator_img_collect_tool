import argparse

import pyrealsense2 as rs
import numpy as np
import cv2



def save_video(video_path, video_fps, video_w, video_h):
    # # VIDEO SETTINGS
    video_name = video_path
    # # REALSENSE SET
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, video_w, video_h, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    # # SAVE
    # fourcc 指定编码器
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Out settings(OpenCV)
    out = cv2.VideoWriter(video_name, fourcc, video_fps, (video_w, video_h), True)
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())  # 转为数组
        cv2.imshow('frame', color_image)
        out.write(color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Shooting Video", add_help=True)
    parser.add_argument("--video_path", type=str, default=r"./output/video.mp4", help="output video file ")
    parser.add_argument("--video_fps", type=int, default=30, help="fps")
    parser.add_argument("--video_w", type=int, default=1920, help="weight")
    parser.add_argument("--video_h", type=int, default=1080, help="height")

    args = parser.parse_args()
    video_path = args.video_path
    video_fps = args.video_fps
    video_w = args.video_w
    video_h = args.video_h
    save_video(video_path, video_fps, video_w, video_h)


