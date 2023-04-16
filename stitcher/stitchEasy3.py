import cv2
import numpy as np
from stitchEasy2 import stitch_images, readCircleParams
from equirec2Perspec import Equirectangular

def main(video_path, left_txt_path, right_txt_path):
    leftX, leftY, leftR = readCircleParams(left_txt_path)
    rightX, rightY, rightR = readCircleParams(right_txt_path)

    cap = cv2.VideoCapture(video_path)

    theta, phi = 0, 0
    fov = 90

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        stitched = stitch_images(frame, leftX, leftY, leftR, rightX, rightY, rightR)

        height, _ = stitched.shape[:2]
        eq = Equirectangular(stitched)
        perspective = eq.GetPerspective(fov, theta, phi, height, height)

        cv2.putText(perspective, f"Longitude: {theta}", (perspective.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(perspective, f"Latitude: {phi}", (perspective.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Perspective Image', perspective)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            phi = min(90, phi + 1)
        elif key == ord('s'):
            phi = max(-90, phi - 1)
        elif key == ord('a'):
            theta = (theta - 1) % 360
        elif key == ord('d'):
            theta = (theta + 1) % 360

    cap.release()
    cv2.destroyAllWindows()

# 示例调用
main('mp4/c2.mp4', 'pics/left.txt', 'pics/right.txt')