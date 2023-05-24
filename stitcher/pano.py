import os
import json

class Pano:
    def __init__(self) -> None:
        self.config_path = './config.json'
        self.config = self.readConfig()
    
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
                config = json.load(config_file)

        return config

    def writeConfig(self):
        config_path = self.config_path
        # 将配置写入配置文件
        with open(config_path, 'w') as config_file:
            json.dump(self.config, config_file)

    

if __name__ == '__main__':
    pano = Pano()
    print(pano.config)