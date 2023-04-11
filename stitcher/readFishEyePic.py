# 输入图片路径，路径类似'pics/Left.jpg'
# 读取指定路径的图片，并在相同目录下，找到相同文件名的txt文件，读取圆心和半径，格式是'{x} {y} {r}'。
# 如果没有找到相同文件名的txt文件，就使用opencv去找默认的圆心和半径。
# 在窗口中显示图片，并用显著的颜色画出圆形和半径。
# 用键盘调整圆心和半径，A,S,W,D，移动圆心，Q,E，调整半径。按照图片尺寸调整步长，每次0.5%。
# 调整好之后输入回车，保存圆心和半径到txt文件。
# 最后按照圆心和半径裁剪图片，裁剪成一个与圆形外切的正方形，超出画面的部分用黑色填充。
# 返回裁剪后的图片。展示在窗口中。按回车退出

import cv2 
import numpy as np 
import os

# 修改readFishEyePic，加入参数 needShow，默认为False
# 如果needShow为False，就不再需要调整图片的圆心和半径，也不再显示图片到窗口
# 如果needShow为False并且并未找到相同文件名的txt文件，就报错，提示必须先执行此函数并且needShow = Ture进行调整

def readFishEyePic(image_path): 
    def detect_circle_center(image): 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        blur = cv2.GaussianBlur(gray, (5, 5), 0) 
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0) 
        if circles is not None: 
            circles = np.round(circles[0, :]).astype("int") 
            x, y, r = circles[0] 
            return x, y, r 
        return None

    def draw_circle_center(image, circle_center):
        x, y, r = circle_center
        image_with_center = image.copy()
        cv2.circle(image_with_center, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(image_with_center, (x, y), r, (0, 255, 0), 2)
        return image_with_center

    def crop_image(image, circle_center):
        x, y, r = circle_center
        h, w, _ = image.shape
        top = max(0, y - r)
        bottom = min(h, y + r)
        left = max(0, x - r)
        right = min(w, x + r)
        cropped_image = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
        cropped_image[top - (y - r):bottom - (y - r), left - (x - r):right - (x - r)] = image[top:bottom, left:right]
        return cropped_image

    # 读取图片
    image = cv2.imread(image_path)
    base_path, file_name = os.path.split(image_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    txt_path = os.path.join(base_path, file_name_without_ext + '.txt')

    # 读取圆心和半径
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            x, y, r = map(int, f.readline().strip().split())
    else:
        circle_center = detect_circle_center(image)
        if circle_center is not None:
            x, y, r = circle_center
        else:
            print("Error: Could not detect circle center.")
            return

    step = int(0.005 * max(image.shape[:2]))
    while True:
        # 在图像上绘制圆心和圆
        image_with_center = draw_circle_center(image, (x, y, r))

        # 显示绘制圆心和圆后的图像
        cv2.imshow('Image with Circle Center and Circle', image_with_center)

        # 获取键盘输入
        key = cv2.waitKey(0) & 0xFF

        # 根据键盘输入调整圆心和半径
        if key == ord('a'):
            x -= step
        elif key == ord('s'):
            y += step
        elif key == ord('w'):
            y -= step
        elif key == ord('d'):
            x += step
        elif key == ord('q'):
            r -= step
        elif key == ord('e'):
            r += step
        elif key == 13:  # 回车键
            break

    # 保存圆心和半径到文件
    with open(txt_path, 'w') as f:
        f.write(f'{x} {y} {r}\n')

    # 裁剪图片
    cropped_image = crop_image(image, (x, y, r))

    # 显示裁剪后的图片
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_image

if __name__ == '__main__': 
    # img = readFishEyePic('pics/Left.jpg')
    # cv2.imwrite('pics/croppedLeft.jpg', img)

    img = readFishEyePic('pics/21.jpg')
    cv2.imwrite('pics/cropped21.jpg', img)
