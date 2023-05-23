import cv2
import numpy as np

def draw_matches(img1, img2, control_points1, control_points2, overlap_width):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    output = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    output[0:h1, 0:w1] = img1
    output[0:h2, w1:] = img2

    for pt1, pt2 in zip(control_points1, control_points2):
        x1, y1 = map(int, pt1)
        x2, y2 = map(int, pt2)

        # Generate a random color for each pair of matched points
        color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.circle(output, (x1, y1), 5, color, 1)
        cv2.circle(output, (x2 + w1, y2), 5, color, 1)
        cv2.line(output, (x1, y1), (x2 + w1, y2), color, 1)

    # 绘制半透明的重叠区域矩形
    overlay = output.copy()
    cv2.rectangle(overlay, (w1 - overlap_width, 0), (w1, max(h1, h2)), (0, 255, 0), -1)
    cv2.rectangle(overlay, (w1, 0), (w1 + overlap_width, max(h1, h2)), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

    return output


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

def filter_homography(keypoints1, keypoints2, matches, threshold=5.0):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    matches_mask = mask.ravel().tolist()

    filtered_matches = [m for i, m in enumerate(matches) if matches_mask[i]]

    return filtered_matches

# 高度差距过大的点不要
def filter_height_diff(control_points1, control_points2, img1_full, img2_full):
    filtered_control_points1 = []
    filtered_control_points2 = []
    for pt1, pt2 in zip(control_points1, control_points2):
        if abs(pt1[1] - pt2[1]) < img1_full.shape[0] * 0.02:
            filtered_control_points1.append(pt1)
            filtered_control_points2.append(pt2)

    return filtered_control_points1, filtered_control_points2

def filter_edge_points(control_points1, control_points2, img1_full, img2_full):
    filtered_control_points1 = []
    filtered_control_points2 = []
    for pt1, pt2 in zip(control_points1, control_points2):
        if pt1[0] > img1_full.shape[1] * 0.9 and pt2[0] < img2_full.shape[1] * 0.1:
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

    return control_points1, control_points2

def visualize_homography(img1, img2, M, is_affine=False):
    corners_img1 = np.float32([[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]]).reshape(-1, 1, 2)
    
    if is_affine:
        corners_transformed = cv2.transform(corners_img1, M)
    else:
        corners_transformed = cv2.perspectiveTransform(corners_img1, M)
    
    img2_corners_visualized = img2.copy()
    for i in range(4):
        x1, y1 = map(int, corners_transformed[i, 0])
        x2, y2 = map(int, corners_transformed[(i + 1) % 4, 0])
        cv2.line(img2_corners_visualized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img2_corners_visualized



def main():
    img2_full = cv2.imread('pics2/230514_104354LeftEC.jpg')
    img1_full = cv2.imread('pics2/230514_104354RightEC.jpg')
    print('img1_full.shape:',img1_full.shape)
    print('img2_full.shape:',img2_full.shape)


    control_points1, control_points2 = get_control_points_flann(img1_full, img2_full)

    # print('control_points1 len:',len(control_points1))
    # print('control_points1:',control_points1)
    # print('control_points2:',control_points2)

    # 计算重叠区域的宽度
    overlap_width = img1_full.shape[1] - int(min(control_points1, key=lambda x: x[0])[0])

    # 绘制匹配的特征点
    matched_points_img = draw_matches(img1_full, img2_full, control_points1[:4], control_points2[:4], overlap_width)
    cv2.imwrite('pics2/stitched21_mid.jpg', matched_points_img)

    control_points1 = np.float32(control_points1)
    control_points2 = np.float32(control_points2)

    # 这里自己写一个平移的矩阵变换
    # 变换的规则是按照control_points2 -> control_points1 进行变换
    # 先获得平移的平均向量 这个向量命名为dv。此dv是cp2针对cp1的向量。cp2 + dv = cp1 => dv = cp1 - cp2
    # 所以，先计算cp1 - cp2的平均值
    dv = np.mean(control_points1 - control_points2, axis=0)
    print('control_points1:',control_points1)
    print('control_points2:',control_points2)
    print('dv:',dv)

    # 坐标变换，将img2_full的坐标变换到img1_full的坐标
    # 即img2_full + dv = img1_full
    # 创建一个2x3的仿射变换矩阵，其中平移向量在最后一列
    affine_matrix = np.float32([[1, 0, dv[0]], [0, 1, dv[1]]])

    # 应用仿射变换矩阵
    img2_translated = cv2.warpAffine(img2_full, affine_matrix, (img1_full.shape[1] + img2_full.shape[1]-overlap_width, img1_full.shape[0]))

    result = img2_translated.copy()
    # 用img2_translated作为底图
    # 底图是经过变换后的img2，应该在画面右侧
    # 将画面左侧，0~img1_full.shape[1]-overlap_width的部分，设置为img1_full
    # 中间部分即从img1_full.shape[1]-overlap_width~img1_full.shape[1]，设置为img1_full和img2_translated的权重融合
    result[:, :img1_full.shape[1] - overlap_width] = img1_full[:, :img1_full.shape[1] - overlap_width]

    # 对于重叠区域，使用加权融合方法
    # for i in range(overlap_width):
    #     # 计算权重
    #     alpha = float(i) / overlap_width
    #     beta = 1.0 - alpha

    #     # 对于每个像素，按权重融合两个图像
    #     result[:, img1_full.shape[1] - overlap_width + i] = (alpha * img1_full[:, img1_full.shape[1] - overlap_width + i] +
    #                                                         beta * img2_translated[:, img1_full.shape[1] - overlap_width + i]).astype(np.uint8)

    # 显示结果
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
