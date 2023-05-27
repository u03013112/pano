# 尝试用tk做一个好看一点的UI
# 先做一个简单的demo
# 将主窗口分为两个部分，左边是一些文字信息和控制按钮，右侧是视频展示区
# 用cv2打开'../mp4/c2.mp4'
# 视频信息显示在左侧，包括分辨率，帧率，时长，目前播放进度（默认从头开始）
# 左侧添加控制按钮，包括播放，暂停，重新开始3个按钮
# 右侧添加视频展示区，用于展示视频
# 整个窗口的高度取自视频的高度，宽度取自视频的宽度+左侧信息栏的宽度
# 播放部分，用cv2读取视频，然后逐帧处理成tk可以显示的格式，然后显示在右侧的视频展示区

import cv2
import tkinter as tk
from PIL import Image, ImageTk

# 打开视频
video_path = '../mp4/c2.mp4'
video = cv2.VideoCapture(video_path)

# 获取视频信息
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

# 创建Tkinter窗口
root = tk.Tk()
root.geometry(f'{width + 200}x{height}')
root.title('Video Player')

# 创建左侧信息栏
info_frame = tk.Frame(root, width=200, height=height, bg='white')
info_frame.pack(side=tk.LEFT, fill=tk.Y)

# 显示视频信息
info_text = f'分辨率: {width}x{height}\n帧率: {fps}\n时长: {duration}s\n播放进度: 0s'
info_label = tk.Label(info_frame, text=info_text, bg='white', justify=tk.LEFT)
info_label.pack(pady=10)

# 创建控制按钮
is_playing = True

def play():
    global is_playing
    is_playing = True

def pause():
    global is_playing
    is_playing = False

def restart():
    global is_playing
    is_playing = True
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

play_button = tk.Button(info_frame, text='播放', command=play)
play_button.pack(pady=10)

pause_button = tk.Button(info_frame, text='暂停', command=pause)
pause_button.pack(pady=10)

restart_button = tk.Button(info_frame, text='重新开始', command=restart)
restart_button.pack(pady=10)

# 创建右侧视频展示区
video_frame = tk.Frame(root, width=width, height=height)
video_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

video_label = tk.Label(video_frame)
video_label.pack(fill=tk.BOTH)

# 播放视频
def updatePer33Ms():
    global is_playing
    if is_playing:
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

            # 更新播放进度
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = int(current_frame / fps)
            info_label.config(text=f'分辨率: {width}x{height}\n帧率: {fps}\n时长: {duration}s\n播放进度: {current_time}s')

    video_label.after(33, updatePer33Ms)

updatePer33Ms()
root.mainloop()

# 释放资源
video.release()
cv2.destroyAllWindows()
