# 圆形鱼眼图片
# 输入是图片，readFishEyePic返回值，一个正方的图片，与圆形外切，超出画面的部分用黑色填充。
# 用opencv的buildMap和remap方式，把圆形鱼眼图片转换成等距投影图。
# 可输入参数：鱼眼视角（上下左右都是这个角度）
# 不在输入参数：鱼眼图片宽高，这个由输入图片决定
# 不在输入参数：输出图片宽高，这个由输入图片和鱼眼视角参数决定
# 将转换完的图片展示在窗口中，按回车退出
# 返回值：转换完的图片

import cv2
import numpy as np

def equirect_proj(x_proj, y_proj, W, H, fov):
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * np.pi / H

    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = np.cos(theta_alt) * np.cos(phi_alt)

    return np.arctan2(y, x), np.arctan2(np.sqrt(x**2 + y**2), z)

def buildmap(Ws, Hs, Wd, Hd, fov=180.0):
    fov = fov * np.pi / 180.0

    ys, xs = np.indices((Hs, Ws), np.float32)
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0

    theta, phi = equirect_proj(x_proj, y_proj, Ws, Hs, fov)

    p = Hd * phi / fov

    y_fish = p * np.sin(theta)
    x_fish = p * np.cos(theta)

    ymap = Hd / 2.0 - y_fish
    xmap = Wd / 2.0 + x_fish
    return xmap, ymap

# 修改fisheye2Equirectangular，添加参数needShow，默认是False
# 如果needShow是False，就不再显示图片
def fisheye2Equirectangular(fisheye_image, fov_degrees, needShow=False):
    src_height, src_width, _ = fisheye_image.shape
    dst_width = src_width
    dst_height = src_height

    map_x, map_y = buildmap(src_width, src_height, dst_width, dst_height, fov_degrees)
    equirectangular_image = cv2.remap(fisheye_image, map_x, map_y, cv2.INTER_LINEAR)

    if needShow:
        cv2.imshow('Equirectangular Image', equirectangular_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return equirectangular_image

# 为了提高效率，把上面fisheye2Equirectangular拆分成两个函数
def buildMap(fisheye_image, fov_degrees):
    src_height, src_width, _ = fisheye_image.shape
    dst_width = src_width
    dst_height = src_height

    map_x, map_y = buildmap(src_width, src_height, dst_width, dst_height, fov_degrees)
    return map_x, map_y

def remap(fisheye_image, map_x, map_y):
    equirectangular_image = cv2.remap(fisheye_image, map_x, map_y, cv2.INTER_LINEAR)
    return equirectangular_image

import time
def fisheye2EquirectangularDebug(fisheye_image, fov_degrees, needShow=False):
    src_height, src_width, _ = fisheye_image.shape
    dst_width = src_width
    dst_height = src_height

    start_time = time.time()
    map_x, map_y = buildmap(src_width, src_height, dst_width, dst_height, fov_degrees)
    buildmap_time = time.time() - start_time

    start_time = time.time()
    equirectangular_image = cv2.remap(fisheye_image, map_x, map_y, cv2.INTER_LINEAR)
    remap_time = time.time() - start_time

    if needShow:
        cv2.imshow('Equirectangular Image', equirectangular_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("buildmap耗时：{:.2f} ms".format(buildmap_time * 1000))
    print("remap耗时：{:.2f} ms".format(remap_time * 1000))

    return equirectangular_image

import os
import sys
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("错误：请输入图片文件名")
        print("用法：python readFishEyePic.py 图片文件名")
        sys.exit(1)

    img_filename = sys.argv[1]
    base_path, file_name = os.path.split(img_filename)
    file_name_without_ext = os.path.splitext(file_name)[0]
    file_name_ext = os.path.splitext(file_name)[1]
    leftFilePath = os.path.join(base_path, file_name_without_ext + 'Left' + file_name_ext)
    leftOutFilePath = os.path.join(base_path, file_name_without_ext + 'LeftEC' + file_name_ext)
    rightFilePath = os.path.join(base_path, file_name_without_ext + 'Right' + file_name_ext)
    rightOutFilePath = os.path.join(base_path, file_name_without_ext + 'RightEC' + file_name_ext)

    fisheye_image = image = cv2.imread(leftFilePath)
    fov_degrees = 210
    equirectangular_image = fisheye2Equirectangular(fisheye_image, fov_degrees)
    cv2.imwrite(leftOutFilePath,equirectangular_image)

    fisheye_image = image = cv2.imread(rightFilePath)
    fov_degrees = 210
    equirectangular_image = fisheye2Equirectangular(fisheye_image, fov_degrees)
    cv2.imwrite(rightOutFilePath,equirectangular_image)
