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

if __name__ == '__main__':
    fisheye_image = image = cv2.imread('pics/croppedLeft.jpg')
    fov_degrees = 210
    equirectangular_image = fisheye2Equirectangular(fisheye_image, fov_degrees)