# 3段法进行拼接
# 暂时使用这两张图片做测试
# img2_full = cv2.imread('pics2/230514_104354LeftEC.jpg')
# img1_full = cv2.imread('pics2/230514_104354RightEC.jpg')
# 将img2_full拆成2个部分
# 右侧贴在img1_full的左侧，左侧贴在img1_full的右侧
# 拼接方案待定

import cv2
import numpy as np

def find_keypoints_and_matches_flann(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    return keypoints1, keypoints2, matches

def filter_distance(matches, ratio=0.6):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def filter_edge_points(control_points1, control_points2, img1_full, img2_full):
    filtered_control_points1 = []
    filtered_control_points2 = []
    for pt1, pt2 in zip(control_points1, control_points2):
        if pt1[0] > img1_full.shape[1] * 0.9 and pt2[0] < img2_full.shape[1] * 0.1:
            filtered_control_points1.append(pt1)
            filtered_control_points2.append(pt2)

    return filtered_control_points1, filtered_control_points2

# 高度差距过大的点不要
def filter_height_diff(control_points1, control_points2, img1_full, img2_full):
    filtered_control_points1 = []
    filtered_control_points2 = []
    for pt1, pt2 in zip(control_points1, control_points2):
        if abs(pt1[1] - pt2[1]) < img1_full.shape[0] * 0.02:
            filtered_control_points1.append(pt1)
            filtered_control_points2.append(pt2)

    return filtered_control_points1, filtered_control_points2

def get_control_points_flann(img1_full, img2_full):
    keypoints1, keypoints2, matches = find_keypoints_and_matches_flann(img1_full, img2_full)

    # Apply filters
    matches = filter_distance(matches, ratio=0.65)

    control_points1 = []
    control_points2 = []
    for m in matches:
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt

        control_points1.append(pt1)
        control_points2.append(pt2)

    # Uncomment the line below to enable the edge points filter
    control_points1, control_points2 = filter_edge_points(control_points1, control_points2, img1_full, img2_full)
    control_points1, control_points2 = filter_height_diff(control_points1, control_points2, img1_full, img2_full)

    return np.float32(control_points1), np.float32(control_points2)

def translate_image_y(img, y_offset):
    translated_img = np.zeros_like(img)
    if y_offset > 0:
        translated_img[y_offset:, :] = img[:-y_offset, :]
    elif y_offset < 0:
        translated_img[:y_offset, :] = img[-y_offset:, :]
    else:
        translated_img = img.copy()
    return translated_img

# 对步骤进行优化
def stitch2(img1,img2):
    # 将img2分成两部分
    img2_left = img2[:, :int(img2.shape[1] / 2)]
    img2_right = img2[:, int(img2.shape[1] / 2):]

    # 计算 img2_right 和 img1 的特征点
    control_points1, control_points2 = get_control_points_flann(img2_right, img1)
    overlap_width1 = img2_right.shape[1] - int(min(control_points1, key=lambda x: x[0])[0]) + int(min(control_points2, key=lambda x: x[0])[0])
    # 计算 偏移均值left
    dv1 = np.mean(control_points1 - control_points2, axis=0)

    # 计算 img1 和 img2_left 的特征点
    control_points1, control_points2 = get_control_points_flann(img1, img2_left)
    overlap_width2 = img1.shape[1] - int(min(control_points1, key=lambda x: x[0])[0]) + int(min(control_points2, key=lambda x: x[0])[0])
    # 计算 偏移均值right
    dv2 = np.mean(control_points1 - control_points2, axis=0)

    # 尝试对 偏移均值left 和 偏移均值right 进行统一，最好是个轴对称，如果不是会很麻烦，需要调整两个摄像头的物理位置
    # 由于一个是img2_right 和 img1，另一个是 img1 和 img2_left，所以他们的dv[1]是相反数
    # 对dv[1]取平均值，再按照原来的符号赋值给dv[1]
    dv1[1] = -dv1[1]
    avg_dv1 = (dv1[1] + dv2[1]) / 2
    dv1[1] = -avg_dv1
    dv2[1] = avg_dv1

    dh = int(avg_dv1)

    # 左右的接缝宽度暂时不做处理

    # 计算画布的应有尺寸
    # 应有尺寸的宽度应该是 img1.shape[1] + img2.shape[1] - overlap_width1 - overlap_width2
    # 应有尺寸的高度应该是 img1.shape[0]，这里以img1的高度为准
    # 制作一个空的画布
    stitchedResult = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1] - overlap_width1 - overlap_width2, 3), dtype=np.uint8)
    # 先放img1，他的高度是不做修正的
    # 他的位置应该是在img2_right的宽度 - overlap_width1
    stitchedResult[:, img2_right.shape[1] - overlap_width1:img2_right.shape[1] - overlap_width1 + img1.shape[1]] = img1

    # 计算并将img2_right放在左侧
    # 先将img2_right上下平移，平移量为dv1[1]*-1
    img2_right_translated = translate_image_y(img2_right, int(dv1[1]*-1))
    # 将img2_right_translated放在stitchedResult的左侧
    stitchedResult[:, :img2_right.shape[1]] = img2_right_translated[:, :img2_right.shape[1]]

    # 计算并将img2_left放在右侧
    # 先将img2_left上下平移，平移量为dv2[1]
    img2_left_translated = translate_image_y(img2_left, int(dv2[1]))
    # 将img2_left_translated放在stitchedResult的右侧
    stitchedResult[:, img2_right.shape[1] - overlap_width1 + img1.shape[1]-overlap_width2:] = img2_left_translated

    # 渐变权重，对接缝进行模糊处理，左侧接缝
    overlap_start = img2_right.shape[1] - overlap_width1
    overlap_end = img2_right.shape[1]
    # 计算重叠区域的权重矩阵
    weight_matrix = np.linspace(0, 1, overlap_end - overlap_start).reshape(1, -1, 1)
    # 使用权重矩阵对重叠区域进行加权平均
    stitchedResult[:, overlap_start:overlap_end] = weight_matrix * img1[:,:overlap_width1] + (1 - weight_matrix) * img2_right_translated[:, overlap_start:overlap_end]

    # 渐变权重，对接缝进行模糊处理，右侧接缝
    overlap_start = img2_right.shape[1] - overlap_width1 + img1.shape[1]-overlap_width2
    overlap_end = img2_right.shape[1] - overlap_width1 + img1.shape[1]
    # 计算重叠区域的权重矩阵
    weight_matrix = np.linspace(0, 1, overlap_end - overlap_start).reshape(1, -1, 1)
    # 使用权重矩阵对重叠区域进行加权平均
    stitchedResult[:, overlap_start:overlap_end] = weight_matrix * img2_left_translated[:, :overlap_width2] + (1 - weight_matrix) * img1[:, -overlap_width2:]

    # 显示结果
    # cv2.imshow('stitchedResult', stitchedResult)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 结果保存
    # cv2.imwrite('pics2/stitchedResult.jpg', stitchedResult)
    return overlap_width1,overlap_width2,dh

# 不再校准，直接用stitch2的结果进行拼接
# 参数：img1是正面图像，img2是背面图像，争取保证正面图像的质量，如果必要牺牲背面图像的质量
# overlap_width1和overlap_width2是两个接缝的宽度
# dh是高度差，这里强制要求两边高度差一致，就是stitch2种dv[1]的平均值
def stitch3(img1,img2,overlap_width1,overlap_width2,dh):
    # 这里面不再调整两张图片的大小，要求图片大小一致
    # TODO：转等距投影的部分把图片大小调整一致

    # 将img2分成两部分
    img2_left = img2[:, :int(img2.shape[1] / 2)]
    img2_right = img2[:, int(img2.shape[1] / 2):]

    stitchedResult = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1] - overlap_width1 - overlap_width2, 3), dtype=np.uint8)
    # 先放img1，他的高度是不做修正的
    # 他的位置应该是在img2_right的宽度 - overlap_width1
    stitchedResult[:, img2_right.shape[1] - overlap_width1:img2_right.shape[1] - overlap_width1 + img1.shape[1]] = img1

    # 计算并将img2_right放在左侧
    img2_right_translated = translate_image_y(img2_right, dh)
    # 将img2_right_translated放在stitchedResult的左侧
    stitchedResult[:, :img2_right.shape[1]] = img2_right_translated[:, :img2_right.shape[1]]

    # 计算并将img2_left放在右侧
    img2_left_translated = translate_image_y(img2_left, dh)
    # 将img2_left_translated放在stitchedResult的右侧
    stitchedResult[:, img2_right.shape[1] - overlap_width1 + img1.shape[1]-overlap_width2:] = img2_left_translated

    # 渐变权重，对接缝进行模糊处理，左侧接缝
    overlap_start = img2_right.shape[1] - overlap_width1
    overlap_end = img2_right.shape[1]
    # 计算重叠区域的权重矩阵
    weight_matrix = np.linspace(0, 1, overlap_end - overlap_start).reshape(1, -1, 1)
    # 使用权重矩阵对重叠区域进行加权平均
    stitchedResult[:, overlap_start:overlap_end] = weight_matrix * img1[:,:overlap_width1] + (1 - weight_matrix) * img2_right_translated[:, overlap_start:overlap_end]

    # 渐变权重，对接缝进行模糊处理，右侧接缝
    overlap_start = img2_right.shape[1] - overlap_width1 + img1.shape[1]-overlap_width2
    overlap_end = img2_right.shape[1] - overlap_width1 + img1.shape[1]
    # 计算重叠区域的权重矩阵
    weight_matrix = np.linspace(0, 1, overlap_end - overlap_start).reshape(1, -1, 1)
    # 使用权重矩阵对重叠区域进行加权平均
    stitchedResult[:, overlap_start:overlap_end] = weight_matrix * img2_left_translated[:, :overlap_width2] + (1 - weight_matrix) * img1[:, -overlap_width2:]

    # # 显示结果
    # cv2.imshow('stitchedResult', stitchedResult)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # # 结果保存
    # cv2.imwrite('pics2/stitchedResult.jpg', stitchedResult)
    return stitchedResult


def main():
    img1 = cv2.imread('pics2/230514_104354LeftEC.jpg')
    img2 = cv2.imread('pics2/230514_104354RightEC.jpg')
    # 将两张图片的shape调整为一致，按照较小的那张图片的shape进行调整
    if img1.shape[0] < img2.shape[0]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    else:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    overlap_width1,overlap_width2,dh = stitch2(img1,img2)

    img3 = cv2.imread('pics2/230514_104323LeftEC.jpg')
    img4 = cv2.imread('pics2/230514_104323RightEC.jpg')

    if img3.shape[0] < img4.shape[0]:
        img4 = cv2.resize(img4, (img3.shape[1], img3.shape[0]))
    else:
        img3 = cv2.resize(img3, (img4.shape[1], img4.shape[0]))


    stitch3(img3,img4,overlap_width1,overlap_width2,dh)

# main()