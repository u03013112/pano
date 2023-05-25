from u0_stitcher.stitcher import Stitcher
from u0_stitcher.errors import CircleCenterNotCalibratedException, StitchNotCalibratedException
import cv2
import time

def main():
    pano = Stitcher()
    current_fps = 0
    video = cv2.VideoCapture('mp4/c2.mp4')

    if not video.isOpened():
        print("无法打开视频文件")
        return

    # 获取原始视频的帧速率
    raw_fps = int(video.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = video.read()

        if ret:
            try:
                # img = pano.stitch(frame)
                img = pano.stitchDebug(frame)
                # img = frame
            except CircleCenterNotCalibratedException as e:
                print(e)
                pano.calibCircleCenter(frame)
            except StitchNotCalibratedException as e:
                print(e)
                pano.calibStitch(frame)
            else:
                frame_count += 1
                elapsed_time = time.time() - start_time

                # 每秒更新一次帧速率显示
                if elapsed_time >= 1:
                    current_fps = frame_count
                    frame_count = 0
                    start_time = time.time()

                # 将帧速率文本添加到图像
                fps_text = f"FPS: {current_fps}"
                raw_fps_text = f"RAW FPS: {raw_fps}"
                cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, raw_fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('img', img)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
