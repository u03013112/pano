import cv2
import numpy as np

def main():
    # 创建两个320x240的黑色图像
    img1 = np.zeros((240, 320, 3), dtype=np.uint8)
    img2 = np.zeros((240, 320, 3), dtype=np.uint8)

    # 在img1上画一个红色矩形
    cv2.rectangle(img1, (5, 5), (315, 235), (0, 0, 255), 2)

    # 在img2上画一个绿色矩形
    cv2.rectangle(img2, (5, 5), (315, 235), (0, 255, 0), 2)

    # # 计算平移矩阵M
    # M = np.float32([[1, 0, 300], [0, 1, 0]])

    # 定义写死的control_points1和control_points2
    # control_points1 = np.float32([[10, 10], [310, 10], [10, 230], [310, 230]])
    # control_points2 = np.float32([[310, 10], [610, 10], [310, 230], [610, 230]])

    control_points1 = np.float32([[10, 15], [310, 15], [10, 235], [310, 235]])
    control_points2 = np.float32([[310, 10], [610, 9], [310, 240], [610, 240]])


    # 计算透视变换矩阵M
    M = cv2.getPerspectiveTransform(control_points1, control_points2)
    print('M:',M)
    
    # 使用透视变换矩阵将图像1变换到图像2
    img1_transformed = cv2.warpPerspective(img1, M, (img1.shape[1] * 2 - 10, img1.shape[0]))

    # 获取非重叠部分
    non_overlap1 = img1_transformed[:, 320:]

    non_overlap2 = img2[:, :310]

    # 获取重叠部分
    overlap1 = img1_transformed[:, 310:320]

    overlap2 = img2[:, 310:]
    overlap2 = cv2.resize(overlap2, (overlap1.shape[1], overlap1.shape[0]))  # 调整overlap2的大小以匹配overlap1

    # 将重叠部分加权合并
    overlap_result = cv2.addWeighted(overlap1, 0.5, overlap2, 0.5, 0)

    # 按顺序组合非重叠部分和重叠部分
    result = np.hstack((non_overlap2, overlap_result, non_overlap1))

    # 显示结果
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
