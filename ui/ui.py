import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from u0_stitcher.stitcher import Stitcher
from u0_stitcher.errors import CircleCenterNotCalibratedException, StitchNotCalibratedException

class App:
    def __init__(self,w = 1209,h = 764):
        self.pano = Stitcher()

        root = tk.Tk()
        root.title("UI设计")
        root.geometry("%dx%d" % (w,h))

        self.root = root
        self.left_frame = tk.Frame(root, width=302, height=h, bg='pink')
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.info_text = tk.Text(self.left_frame, height=40, width=40)
        self.info_text.pack(pady=5)
        self.info_text.insert(tk.END, "欢迎光临")

        self.select_video_source_btn = tk.Button(self.left_frame, text="选择视频输入源", command=self.select_video_source)
        self.select_video_source_btn.pack(pady=5)

        self.original_video_btn = tk.Button(self.left_frame, text="原始视频", command=self.original_video)
        self.original_video_btn.pack(pady=5)

        self.equal_distance_projection_btn = tk.Button(self.left_frame, text="等距投影", command=self.equal_distance_projection)
        self.equal_distance_projection_btn.pack(pady=5)

        self.equal_angle_projection_btn = tk.Button(self.left_frame, text="等角投影", command=self.equal_angle_projection)
        self.equal_angle_projection_btn.pack(pady=5)

        self.manual_calibration_btn = tk.Button(self.left_frame, text="手动校准", command=self.manual_calibration)
        self.manual_calibration_btn.pack(pady=5)

        self.right_frame = tk.Frame(root, width=967, height=785)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.video_label = tk.Label(self.right_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # self.video_source 是视频输入源，可以是文件或者视频流
        self.video_source = None

        self.video_filepath = None
        self.video_stream = None
        
        # self.video 是播放时读取的视频流对象
        self.video = None 

        # 默认用原始播放
        self.play_type = 'original'

        # 是否需要手动校准
        self.needs_manual_calibration = False

        # 开始调度
        self.updatePer33Ms()

    def updatePerFrame(self):
        if self.video_source == 'file':
            if self.video == None:
                self.video = cv2.VideoCapture(self.video_filepath)        
                # 获取视频信息
                video = self.video
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

                self.append_info_text('\n分辨率：%dx%d'%(width,height))
                self.append_info_text('\n帧率：%d'%fps)
                self.append_info_text('\n时长：%ds'%duration)
        
            video = self.video
            ret, frame = video.read()
            if ret:
                if self.needs_manual_calibration:
                    self.pano.calibStitch(frame)
                    self.append_info_text('\n手动校准完成')
                    self.needs_manual_calibration = False

                if self.play_type == 'original':
                    retFrame = frame
                elif self.play_type == 'equal_distance_projection':
                    # 下面这些except暂未测试，可能会有bug
                    try:
                        retFrame = self.pano.stitch(frame)
                    except CircleCenterNotCalibratedException as e:
                        print(e)
                        self.pano.calibCircleCenter(frame)
                    except StitchNotCalibratedException as e:
                        print(e)
                        self.pano.calibStitch(frame)

                frame = cv2.cvtColor(retFrame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk
            else:
                self.video_source = None
                self.video = None
                self.video_label.config(image=None)
                self.video_label.image = None
                self.update_info_text("视频播放完毕")

    def updatePer33Ms(self):
        # 每秒更新30次，即30fps，这只是一个封装，为了显得干净，请将需要逐帧更新的代码写在updatePerFrame函数中
        self.updatePerFrame()
        self.root.after(33, self.updatePer33Ms)

    def update_info_text(self, new_text):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, new_text)

    def append_info_text(self, new_text):
        self.info_text.insert(tk.END, new_text)


    def select_video_source(self):
        def select_video_file():
            self.video_filepath = filedialog.askopenfilename()
            self.update_info_text(f"选择的文件路径: {self.video_filepath}")
            self.video_source = 'file'

        def select_video_stream():
            def submit_stream():
                self.video_stream = stream_entry.get()
                self.update_info_text(f"视频流: {self.video_stream}")
                stream_top.destroy()
                self.video_source = 'stream'

            stream_top = tk.Toplevel(self.root)
            stream_top.title("输入视频流")
            stream_label = tk.Label(stream_top, text="请输入视频流地址:")
            stream_label.pack(pady=5)
            stream_entry = tk.Entry(stream_top)
            stream_entry.pack(pady=5)
            submit_button = tk.Button(stream_top, text="提交", command=submit_stream)
            submit_button.pack(pady=5)

        source_menu = tk.Menu(self.root, tearoff=0)
        source_menu.add_command(label="视频文件", command=select_video_file)
        source_menu.add_command(label="视频流", command=select_video_stream)
        source_menu.post(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def original_video(self):
        self.play_type = 'original'

    def equal_distance_projection(self):
        self.play_type = 'equal_distance_projection'

    def equal_angle_projection(self):
        self.play_type = 'equal_angle_projection'

    def manual_calibration(self):
        self.needs_manual_calibration = True

    def show(self):
        self.root.mainloop()

if __name__ == "__main__":
    
    app = App()
    app.update_info_text("欢迎光临\n请先选择视频输入源")
    app.show()
