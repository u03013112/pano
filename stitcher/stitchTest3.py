# 两张等距投影图进行拼接，图片如下：
# equirectangular_imageLeft = cv2.imread('pics/equirectangularLeft.jpg')
# equirectangular_imageRight = cv2.imread('pics/equirectangularRight.jpg')
# 使用SIFT算法检测控制点
# 要求只从equirectangularLeft右侧的10%的区域和equirectangularRight左侧10%的区域中检测控制点
# Lowe's Ratio Test 暂时先设置 0.6
# 利用RANSAC去掉一些噪音点，参数先大致定一个，后续再改
# 要求在获得控制点后，将两张图片按照左右并排放好，并将匹配的控制点进行连线，保存为图片‘pics/stitched21_mid.jpg’
# 要求控制点连线尽量水平，如果角度超过10度，就舍弃该组控制点

import cv2
import numpy as np

def find_keypoints_and_matches(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

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

def get_control_points(img1_full, img2_full):
    keypoints1, keypoints2, matches = find_keypoints_and_matches(img1_full, img2_full)

    # Apply filters
    # Uncomment the line below to enable the distance filter
    matches = filter_distance(matches,ratio=0.65)
    # Uncomment the line below to enable the homography filter
    # matches = filter_homography(keypoints1, keypoints2, matches)

    control_points1 = []
    control_points2 = []
    for m in matches:
        if isinstance(m, cv2.DMatch):
            pt1 = keypoints1[m.queryIdx].pt
            pt2 = keypoints2[m.trainIdx].pt
        else:  # In case matches are tuples
            pt1 = keypoints1[m[0].queryIdx].pt
            pt2 = keypoints2[m[0].trainIdx].pt

        control_points1.append(pt1)
        control_points2.append(pt2)

    # Uncomment the line below to enable the edge points filter
    control_points1, control_points2 = filter_edge_points(control_points1, control_points2, img1_full, img2_full)
    control_points1, control_points2 = filter_height_diff(control_points1, control_points2, img1_full, img2_full)

    return control_points1, control_points2

def stitch_images(img1, img2, control_points1, control_points2):
    src_pts = np.float32(control_points1).reshape(-1, 1, 2)
    dst_pts = np.float32(control_points2).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    result = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    # 在重叠区域进行平滑融合
    overlap_width = img1.shape[1] - np.argmax(np.sum(result, axis=(0, 2)) == 0)
    img1_weighted = img1[:, -overlap_width:] * (np.arange(overlap_width, 0, -1) / overlap_width)[:, None, None]
    img2_weighted = result[:, :overlap_width] * (np.arange(1, overlap_width + 1) / overlap_width)[:, None, None]
    result[:, :overlap_width] = cv2.addWeighted(img1_weighted, 1, img2_weighted, 1, 0)

    result[0:img1.shape[0], 0:img1.shape[1] - overlap_width] = img1[:, :img1.shape[1] - overlap_width]

    return result

def draw_matches(img1, img2, control_points1, control_points2):
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

    return output


def main():
    img1_full = cv2.imread('pics/equirectangularLeft.jpg')
    img2_full = cv2.imread('pics/equirectangularRight.jpg')

    control_points1, control_points2 = get_control_points(img1_full, img2_full)
    matched_image = draw_matches(img1_full, img2_full, control_points1, control_points2)
    cv2.imwrite('pics/stitched21_mid.jpg', matched_image)

    stitched_result = stitch_images(img1_full, img2_full, control_points1, control_points2)

    cv2.imwrite('pics/stitched_result.jpg', stitched_result)


if __name__ == "__main__":
    main()