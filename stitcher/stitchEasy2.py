import cv2
import numpy as np
import time

from fisheye2Equirectangular import fisheye2Equirectangular

def readCircleParams(filename):
    with open(filename, 'r') as f:
        x, y, r = map(int, f.readline().split())
    return x, y, r

def cropCircle(image, x, y, r):
    height, width = image.shape[:2]

    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    cropped = cv2.bitwise_and(image, image, mask=mask)

    newImage = np.zeros((2 * r, 2 * r, 3), np.uint8)

    top = max(0, y - r)
    bottom = min(height, y + r)
    left = max(0, x - r)
    right = min(width, x + r)

    newTop = r - (y - top)
    newBottom = r + (bottom - y)
    newLeft = r - (x - left)
    newRight = r + (right - x)

    newImage[newTop:newBottom, newLeft:newRight] = cropped[top:bottom, left:right]

    height, width = newImage.shape[:2]
    return newImage

def resizeToMatchHeight(image1, image2):
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    minHeight = min(height1, height2)

    if height1 > minHeight:
        scaleRatio = minHeight / height1
        newWidth = int(width1 * scaleRatio)
        image1 = cv2.resize(image1, (newWidth, minHeight))
    if height2 > minHeight:
        scaleRatio = minHeight / height2
        newWidth = int(width2 * scaleRatio)
        image2 = cv2.resize(image2, (newWidth, minHeight))
    return image1, image2


def stitch_images(frame, leftX, leftY, leftR, rightX, rightY, rightR):
    height, width = frame.shape[:2]
    mid = width // 2
    leftFrame = frame[:, :mid]
    rightFrame = frame[:, mid:]

    croppedLeft = cropCircle(leftFrame, leftX, leftY, leftR)
    croppedRight = cropCircle(rightFrame, rightX, rightY, rightR)

    croppedLeft, croppedRight = resizeToMatchHeight(croppedLeft, croppedRight)

    equirectangularLeft = fisheye2Equirectangular(croppedLeft, 210)
    equirectangularRight = fisheye2Equirectangular(croppedRight, 210)

    stitched = cv2.hconcat([equirectangularLeft, equirectangularRight])
    return stitched

def main(video_path, left_txt_path, right_txt_path):
    leftX, leftY, leftR = readCircleParams(left_txt_path)
    rightX, rightY, rightR = readCircleParams(right_txt_path)

    cap = cv2.VideoCapture(video_path)
    rawFps = int(cap.get(cv2.CAP_PROP_FPS))

    frameCounter = 0
    startTime = time.time()
    realFps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frameCounter += 1
        elapsedTime = time.time() - startTime

        stitched = stitch_images(frame, leftX, leftY, leftR, rightX, rightY, rightR)

        if elapsedTime >= 1:
            realFps = frameCounter
            frameCounter = 0
            startTime = time.time()

        realFpsText = f"REAL FPS: {realFps}"
        rawFpsText = f"RAW FPS: {rawFps}"
        cv2.putText(stitched, realFpsText, (stitched.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(stitched, rawFpsText, (stitched.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Stitched Image', stitched)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 示例调用
main('mp4/c2.mp4', 'pics/left.txt', 'pics/right.txt')