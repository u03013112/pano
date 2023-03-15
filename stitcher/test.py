import cv2
import imutils

# 将网上找到的图片切割一下，再组合
def prepare():
    p0 = cv2.imread('pics/0.png')
    # p1 = cv2.imread('pics/1.png')

    # 切成3张图，将pic0分成两半，然后按照 p0右，p1，p0左 进行拼接，看效果
    l = int(p0.shape[1]/2)
    p0L = p0[:,:l]
    p0R = p0[:,l:]

    cv2.imwrite('pics/0L.png',p0L)
    cv2.imwrite('pics/0R.png',p0R)

def stitch():
    # p0 = cv2.imread('pics/0R.png')
    # p1 = cv2.imread('pics/1.png')
    # p2 = cv2.imread('pics/0L.png')

    # images = [p0,p1,p2]
    p1 = cv2.imread('pics/Left.jpg')
    p2 = cv2.imread('pics/Right.jpg')
    images = [p1,p2]
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    if status == 0:
        cv2.imwrite('pics/out.jpg', stitched)
    else:
        print("[INFO] image stitching failed ({})".format(status))


stitch()

# 拼接测试感觉不好，特别是针对角度变化较大的图片，等我回家用摄像头试试