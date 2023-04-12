# 引入同文件夹下的两个方法
# 用readFishEyePic读取'pics/21.jpg'和'pics/21-2.jpg'，生成两个图片'pics/cropped21.jpg'和'pics/cropped21-2.jpg'
# 用fisheye2Equirectangular将两个图片转换为等距矩形图，生成两个图片'pics/equirectangular21.jpg'和'pics/equirectangular21-2.jpg'
# 尝试用opencv的全景图拼接方法拼接两个图片，按照21在左，21-2在右的次序，如果不行颠倒次序。
# 生成图片'pics/stitched21.jpg'，如果拼接失败，报错
import cv2
import numpy as np
import os

from readFishEyePic import readFishEyePic
from fisheye2Equirectangular import fisheye2Equirectangular

# 用readFishEyePic读取并裁剪鱼眼图像
cropped_image1 = readFishEyePic('pics/21.jpg')
cropped_image2 = readFishEyePic('pics/21-2.jpg')

# 将裁剪后的鱼眼图像转换为等距矩形图
equirectangular_image1 = fisheye2Equirectangular(cropped_image1, 210)
equirectangular_image2 = fisheye2Equirectangular(cropped_image2, 210)

# 保存等距矩形图像
cv2.imwrite('pics/equirectangular21.jpg', equirectangular_image1)
cv2.imwrite('pics/equirectangular21-2.jpg', equirectangular_image2)

# 创建全景图拼接器
stitcher = cv2.Stitcher_create()

# 尝试拼接等距矩形图像
(status, stitched_image) = stitcher.stitch([equirectangular_image1, equirectangular_image2])

# 如果拼接失败，尝试颠倒图像顺序
if status != 0:
    (status, stitched_image) = stitcher.stitch([equirectangular_image2, equirectangular_image1])

# 如果拼接成功，保存拼接后的全景图像
if status == 0:
    cv2.imwrite('pics/stitched21.jpg', stitched_image)
else:
    print("Error: Stitching failed.")