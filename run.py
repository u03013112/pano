from u0_stitcher.stitcher import Stitcher
from u0_stitcher.errors import CircleCenterNotCalibratedException,StitchNotCalibratedException
import cv2

def main():
    pano = Stitcher()
    # img = cv2.imread('pics/230514_104354.jpg')

    # 用opencv打开视频 mp4/c2.mp4
    # 循环读取每一帧图片
    # 打开视频文件
    video = cv2.VideoCapture('mp4/c2.mp4')

    # 检查视频文件是否成功打开
    if not video.isOpened():
        print("无法打开视频文件")
        return

    # 循环读取每一帧
    while True:
        # 读取下一帧，返回值为布尔值（表示是否成功读取）和帧本身
        ret, frame = video.read()

        # 如果成功读取帧，则处理帧
        if ret:
            try:
                img = pano.stitch(frame)
            except CircleCenterNotCalibratedException as e:
                print(e)
                pano.calibCircleCenter(frame)
            except StitchNotCalibratedException as e:
                print(e)
                pano.calibStitch(frame)
            else:
                # 在此处处理帧，例如显示帧
                cv2.imshow('img', img)

                # 按下'q'键退出循环
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
