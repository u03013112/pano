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


def stitch(img1,img2):
    # 将img2分成两部分
    img2_left = img2[:, :int(img2.shape[1] / 2)]
    img2_right = img2[:, int(img2.shape[1] / 2):]

    # 计算img2_right 和 img1 的特征点
    control_points1, control_points2 = get_control_points_flann(img2_right, img1)
    overlap_width1 = img2_right.shape[1] - int(min(control_points1, key=lambda x: x[0])[0])
    # print('control_points1:',control_points1)
    # print('control_points2:',control_points2)

    dv1 = np.mean(control_points1 - control_points2, axis=0)
    affine_matrix1 = np.float32([[1, 0, dv1[0]], [0, 1, dv1[1]]])
    img1_translated = cv2.warpAffine(img1, affine_matrix1, (img2_right.shape[1] + img1.shape[1]-overlap_width1, img2_right.shape[0]))

    img1_translated[:, :img2_right.shape[1] - overlap_width1] = img2_right[:, :img2_right.shape[1] - overlap_width1]

    # 计算img1_translated 和 img2_left 的特征点
    control_points1, control_points2 = get_control_points_flann(img1_translated, img2_left)
    overlap_width2 = img1_translated.shape[1] - int(min(control_points1, key=lambda x: x[0])[0])
    # print('control_points1:',control_points1)
    # print('control_points2:',control_points2)

    dv2 = np.mean(control_points1 - control_points2, axis=0)
    affine_matrix2 = np.float32([[1, 0, dv2[0]], [0, 1, dv2[1]]])
    img2_translated = cv2.warpAffine(img2_left, affine_matrix2, (img2_left.shape[1] + img1_translated.shape[1]-overlap_width2, img2_left.shape[0]))

    img2_translated[:, :img1_translated.shape[1] - overlap_width2] = img1_translated[:, :img1_translated.shape[1] - overlap_width2]

    cv2.imshow('img2_translated', img2_translated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    img1 = cv2.imread('pics2/230514_104354LeftEC.jpg')
    img2 = cv2.imread('pics2/230514_104354RightEC.jpg')
    stitch(img1,img2)


main()