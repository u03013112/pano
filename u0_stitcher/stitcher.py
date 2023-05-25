import os
import cv2 
import json

import numpy as np
from .fisheye2Equirectangular import fisheye2Equirectangular,fisheye2EquirectangularDebug
from .fisheye2Equirectangular import buildMap,remap
from .stitchEasy3 import stitch2,stitch3

from .errors import CircleCenterNotCalibratedException, StitchNotCalibratedException

# 其他导入...

# 您的主文件代码...

def calibCircleCenter(image,x,y,r):
    # 步长暂定为图像宽高的0.5%
    step = int(0.005 * max(image.shape[:2]))

    def draw_circle_center(image, circle_center):
        x, y, r = circle_center
        image_with_center = image.copy()
        cv2.circle(image_with_center, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(image_with_center, (x, y), r, (0, 255, 0), 2)
        return image_with_center
    
    while True:
        # 在图像上绘制圆心和圆
        image_with_center = draw_circle_center(image, (x, y, r))

        # 显示绘制圆心和圆后的图像
        cv2.imshow('Image with Circle Center and Circle', image_with_center)

        # 获取键盘输入
        key = cv2.waitKey(0) & 0xFF

        # 根据键盘输入调整圆心和半径
        if key == ord('a'):
            x -= step
        elif key == ord('s'):
            y += step
        elif key == ord('w'):
            y -= step
        elif key == ord('d'):
            x += step
        elif key == ord('q'):
            r -= step
        elif key == ord('e'):
            r += step
        elif key == 13:  # 回车键
            break

    return int(x),int(y),int(r)

# 按照配置中的圆心和半径进行切割，将两张图片切割成两个矩形，超出原有图像的部分用黑色填充
def crop_image(image, circle_center):
    x, y, r = circle_center
    h, w, _ = image.shape
    top = max(0, y - r)
    bottom = min(h, y + r)
    left = max(0, x - r)
    right = min(w, x + r)
    cropped_image = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    cropped_image[top - (y - r):bottom - (y - r), left - (x - r):right - (x - r)] = image[top:bottom, left:right]
    return cropped_image

class Stitcher:
    def __init__(self) -> None:
        self.config_path = './config.json'
        self.config = self.readConfig()
        self.tmpConfig = {}
    
    def readConfig(self):
        # 读取配置文件，配置文件在 './config.json' 中
        # 如果文件不存在，那么就初始化一个配置文件
        # 如果文件存在，那么就读取配置文件
        # 返回一个字典
        # 初始值 就是一个空字典
        # 目前该字典中包含的键值对有：
        # img1 = 'left',这里left指的是获得的拼接图像的左侧部分，也就是img2是右侧部分
        # img1CircleCenter = [x,y],这里的0,0指的是圆心的坐标,注意是切分成两个图像后的圆心坐标
        # img2CircleCenter = [x,y],这里的0,0指的是圆心的坐标,注意是切分成两个图像后的圆心坐标
        # imgCircleRadius = 0,这里的r指的是圆的半径，两个圆的半径要一致
        # overlap_width1 = 0,
        # overlap_width2 = 0
        # dh = 0
        config_path = self.config_path

        # 如果配置文件不存在，初始化一个空字典作为配置
        if not os.path.exists(config_path):
            config = {}
            with open(config_path, 'w') as config_file:
                json.dump(config, config_file)
        else:
            # 如果配置文件存在，读取配置文件
            with open(config_path, 'r') as config_file:
                # 尝试反序列化配置文件，如果不能有效的将配置文件反序列化，那么就警告
                # 并将配置文件初始化为空字典
                try:
                    config = json.load(config_file)
                except:
                    print('警告：配置文件格式错误，将初始化配置文件')
                    config = {}
                    with open(config_path, 'w') as config_file:
                        json.dump(config, config_file)

        return config

    def writeConfig(self):
        config_path = self.config_path
        # 将配置写入配置文件
        with open(config_path, 'w') as config_file:
            json.dump(self.config, config_file, indent=4)

    def calibMainCamera(self,mainCamera = 'left'):
        # mainCamera参数必须是'left'或者'right'，否则会报错
        if mainCamera != 'left' and mainCamera != 'right':
            raise Exception('mainCamera参数必须是left或者right')
        
        self.config['img1'] = mainCamera
        # 我的相机是210度的，所以fov是210，这里暂时写死
        self.config['fov'] = 210
        self.writeConfig()

    def calibCircleCenter(self,img):
        # img 是左右拼接的图片，每次校准要两个一起校准
        # 将图片分开
        img1 = img[:, :int(img.shape[1] / 2)]
        img2 = img[:, int(img.shape[1] / 2):]

        # 如果配置文件中有圆心和半径的配置，那么就读取配置
        # 否则要自动探测，这个不太准，聊胜于无
        if 'img1CircleCenter' in self.config and 'img2CircleCenter' in self.config and 'imgCircleRadius' in self.config:
            x1,y1 = self.config['img1CircleCenter']
            x2,y2 = self.config['img2CircleCenter']
            r1 = r2 = self.config['imgCircleRadius']
        else:
            def detect_circle_center(image): 
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                blur = cv2.GaussianBlur(gray, (5, 5), 0) 
                circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0) 
                if circles is not None: 
                    circles = np.round(circles[0, :]).astype("int") 
                    x, y, r = circles[0] 
                    return x, y, r 
                return None
            x1,y1,r1 = detect_circle_center(img1)
            x2,y2,r2 = detect_circle_center(img2)

        x1,y1,r1 = calibCircleCenter(img1,x1,y1,r1)
        x2,y2,r2 = calibCircleCenter(img2,x2,y2,r2)

        r = int((r1+r2)/2)
        # 写配置
        self.config['img1CircleCenter'] = [x1,y1]
        self.config['img2CircleCenter'] = [x2,y2]
        self.config['imgCircleRadius'] = r
        self.writeConfig()

    def buildMap(self,img1):
        if 'mapX' not in self.tmpConfig:
            self.tmpConfig['mapX'],self.tmpConfig['mapY'] = buildMap(img1,self.config['fov'])
        return self.tmpConfig['mapX'],self.tmpConfig['mapY']
    def fisheye2Equirectangular(self,img1):
        mapX,mapY = self.buildMap(img1)
        return remap(img1,mapX,mapY)

    def calibStitch(self,img):
        # 进行判断，
        # 如果配置文件中没有img1，则默认左侧是img1
        # 如果配置文件中没有圆心和半径的配置，那么就报错，调用者收到这个报错应该调用calibCircleCenter
        if 'img1' not in self.config:
            self.calibMainCamera()
        if 'img1CircleCenter' not in self.config or 'img2CircleCenter' not in self.config or 'imgCircleRadius' not in self.config:
            raise CircleCenterNotCalibratedException('请先校准圆心和半径，调用calibCircleCenter')
        
        # 将图片分开
        img1 = img[:, :int(img.shape[1] / 2)]
        img2 = img[:, int(img.shape[1] / 2):]
        
        img1Croped = crop_image(img1,(self.config['img1CircleCenter'][0],self.config['img1CircleCenter'][1],self.config['imgCircleRadius']))
        img2Croped = crop_image(img2,(self.config['img2CircleCenter'][0],self.config['img2CircleCenter'][1],self.config['imgCircleRadius']))


        # 将两张方图分别做成等距投影
        img1EC = self.fisheye2Equirectangular(img1Croped)
        img2EC = self.fisheye2Equirectangular(img2Croped)

        overlap_width1,overlap_width2,dh = stitch2(img1EC,img2EC)
        # 写配置文件
        self.config['overlap_width1'] = overlap_width1
        self.config['overlap_width2'] = overlap_width2
        self.config['dh'] = dh
        self.writeConfig()

    def stitch(self,img):
        # 进行判断，
        # 如果配置文件中没有img1，则默认左侧是img1
        # 如果配置文件中没有圆心和半径的配置，那么就报错，调用者收到这个报错应该调用calibCircleCenter
        if 'img1' not in self.config:
            self.calibMainCamera()
        if 'img1CircleCenter' not in self.config or 'img2CircleCenter' not in self.config or 'imgCircleRadius' not in self.config:
            raise CircleCenterNotCalibratedException('请先校准圆心和半径，调用calibCircleCenter')
        # 如果配置文件中没有overlap_width1，overlap_width2，dh的配置，那么就报错，调用者收到这个报错应该调用calibStitch
        if 'overlap_width1' not in self.config or 'overlap_width2' not in self.config or 'dh' not in self.config:
            raise StitchNotCalibratedException('请先校准拼接，调用calibStitch')
        # 将图片分开
        img1 = img[:, :int(img.shape[1] / 2)]
        img2 = img[:, int(img.shape[1] / 2):]

        img1Croped = crop_image(img1,(self.config['img1CircleCenter'][0],self.config['img1CircleCenter'][1],self.config['imgCircleRadius']))
        img2Croped = crop_image(img2,(self.config['img2CircleCenter'][0],self.config['img2CircleCenter'][1],self.config['imgCircleRadius']))


        # 将两张方图分别做成等距投影
        img1EC = self.fisheye2Equirectangular(img1Croped)
        img2EC = self.fisheye2Equirectangular(img2Croped)

        stitchedResult = stitch3(img1EC,img2EC,self.config['overlap_width1'],self.config['overlap_width2'],self.config['dh'])

        return stitchedResult

    def stitchDebug(self, img):
        import time
        start_time = time.time()

        if 'img1' not in self.config:
            self.calibMainCamera()
        if 'img1CircleCenter' not in self.config or 'img2CircleCenter' not in self.config or 'imgCircleRadius' not in self.config:
            raise CircleCenterNotCalibratedException('请先校准圆心和半径，调用calibCircleCenter')
        if 'overlap_width1' not in self.config or 'overlap_width2' not in self.config or 'dh' not in self.config:
            raise StitchNotCalibratedException('请先校准拼接，调用calibStitch')

        step1_time = time.time()

        img1 = img[:, :int(img.shape[1] / 2)]
        img2 = img[:, int(img.shape[1] / 2):]

        step2_time = time.time()

        img1Croped = crop_image(img1, (self.config['img1CircleCenter'][0], self.config['img1CircleCenter'][1], self.config['imgCircleRadius']))
        img2Croped = crop_image(img2, (self.config['img2CircleCenter'][0], self.config['img2CircleCenter'][1], self.config['imgCircleRadius']))

        step3_time = time.time()

        img1EC = self.fisheye2Equirectangular(img1Croped)
        img2EC = self.fisheye2Equirectangular(img2Croped)

        step4_time = time.time()

        stitchedResult = stitch3(img1EC, img2EC, self.config['overlap_width1'], self.config['overlap_width2'], self.config['dh'])

        step5_time = time.time()

        print("Step 1 (calibration checks) duration: {:.4f} seconds".format(step1_time - start_time))
        print("Step 2 (split image) duration: {:.4f} seconds".format(step2_time - step1_time))
        print("Step 3 (crop image) duration: {:.4f} seconds".format(step3_time - step2_time))
        print("Step 4 (fisheye to equirectangular) duration: {:.4f} seconds".format(step4_time - step3_time))
        print("Step 5 (stitching) duration: {:.4f} seconds".format(step5_time - step4_time))
        print("All duration: {:.4f} seconds".format(step5_time - start_time))

        return stitchedResult


    

if __name__ == '__main__':
    pano = Stitcher()
    # pano.calibMainCamera('left')
    # pano.calibCircleCenter(cv2.imread('pics2/230514_104323.jpg'))
    # print(pano.config)
    # pano.calibStitch(cv2.imread('pics2/230514_104354.jpg'))
    img = pano.stitch(cv2.imread('../pics/230514_104354.jpg'))
    print(img.shape)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
