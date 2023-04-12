import cv2
import numpy as np

def stitch_images(img1, img2):
    # 将图像转换为灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建ORB特征检测器
    orb = cv2.ORB_create()

    # 检测特征点和计算描述符
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 匹配特征描述符
    matches = bf.match(des1, des2)

    # 根据距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 选择最佳匹配
    num_matches = 10
    good_matches = matches[:num_matches]

    # 计算单应性矩阵
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 拼接图像
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result

# 读取图像
equirectangular_image1 = cv2.imread('pics/equirectangular21.jpg')
equirectangular_image2 = cv2.imread('pics/equirectangular21-2.jpg')

# 拼接图像
stitched_image = stitch_images(equirectangular_image1, equirectangular_image2)

# 保存拼接后的全景图像
cv2.imwrite('pics/stitched21.jpg', stitched_image)