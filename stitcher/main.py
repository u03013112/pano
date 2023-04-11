from readFishEyePic import readFishEyePic
from fisheye2Equirectangular import fisheye2Equirectangular

# 引入同文件夹下的两个方法
# 用readFishEyePic读取'pics/21.jpg'和'pics/21-2.jpg'，生成两个图片'pics/cropped21.jpg'和'pics/cropped21-2.jpg'
# 用fisheye2Equirectangular将两个图片转换为等距矩形图，生成两个图片'pics/equirectangular21.jpg'和'pics/equirectangular21-2.jpg'
# 尝试用opencv的全景图拼接方法拼接两个图片，按照21在左，21-2在右的次序，如果不行颠倒次序。
# 生成图片'pics/stitched21.jpg'，如果拼接失败，报错