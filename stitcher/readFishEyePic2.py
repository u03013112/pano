# 与readFishEyePic的区别是，这是读取一张拼接图片
# 先将图片平均分为左右两部分，然后再分别进行readFishEyePic逻辑
# 最终保存文件名为 输入文件名+ 'Left' / 'Right'

import cv2 
import numpy as np 
import os

# 修改readFishEyePic，加入参数 needShow，默认为False
# 如果needShow为False，就不再需要调整图片的圆心和半径，也不再显示图片到窗口
# 如果needShow为False并且并未找到相同文件名的txt文件，就报错，提示必须先执行此函数并且needShow = Ture进行调整

def readFishEyePic(image_path, needShow=False):
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
        if needShow:
            circle_center = detect_circle_center(image)
            if circle_center is not None:
                x, y, r = circle_center
            else:
                print("Error: Could not detect circle center.")
                return
        else:
            raise FileNotFoundError("Error: Could not find the txt file with circle center and radius. Please run this function with needShow=True to adjust the parameters first.")

    if needShow:
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

    if needShow:
        # 显示裁剪后的图片
        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_image

def split_image(image):
    height, width, _ = image.shape
    left_image = image[:, :width // 2]
    right_image = image[:, width // 2:]
    return left_image, right_image

import sys
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("错误：请输入图片文件名")
        print("用法：python readFishEyePic.py 图片文件名")
        sys.exit(1)

    img_filename = sys.argv[1]
    img = cv2.imread(img_filename)
    left_img, right_img = split_image(img)

    base_path, file_name = os.path.split(img_filename)
    file_name_without_ext = os.path.splitext(file_name)[0]
    file_name_ext = os.path.splitext(file_name)[1]
    leftFilePath = os.path.join(base_path, file_name_without_ext + 'Left' + file_name_ext)
    rightFilePath = os.path.join(base_path, file_name_without_ext + 'Right' + file_name_ext)

    cv2.imwrite(leftFilePath, left_img)
    cv2.imwrite(rightFilePath, right_img)

    left_fisheye_img = readFishEyePic(leftFilePath,True)
    right_fisheye_img = readFishEyePic(rightFilePath,True)

    cv2.imwrite(leftFilePath, left_fisheye_img)
    cv2.imwrite(rightFilePath, right_fisheye_img)