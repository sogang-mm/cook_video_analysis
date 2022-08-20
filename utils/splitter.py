# -*- coding: utf-8 -*-

import os
import cv2
from pathlib import Path

def splitter(video_path, segment_sec, sav_dir, fourcc):
    
    """
    1. video_path : str, 구간 분할할 영상 경로 e.g) '/home/Cook_Video_Analysis/Dataset/meat.mp4'
    2. segment_sec: int, 몇 초씩 구간 분할을 할 것인지 e.g) 10
    3. sav_dir: str, 저장할 경로 e.g) '/home/Cook_Video_Analysis/Dataset/segment/'
    4. fourcc: str, four character code : 코덱, 압축 방식, 색상, 픽셀 포맷 등을 정의하는 정수 값 e.g) 'mp4v'
    """
    
    vc = cv2.VideoCapture(video_path)
    
    # check whether video can be opened or not
    if not vc.isOpened():

        print("fail to open video")

    else:
        
        # read first frame of video
        ret, frame = vc.read()
        h, w, _ = frame.shape

        fps = round(vc.get(cv2.CAP_PROP_FPS))
        # step: frame number of 1 segment
        step = segment_sec * fps
        frame_num = round(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        out_frame_num = frame_num // step + 1 # +1: 나머지 frame을 1개로 추가
        
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        
        # save directory
        os.makedirs(f"{sav_dir}{Path(video_path).stem}", exist_ok=True)
        
        # how to write video
        video_name = Path(video_path).stem
        writers = [cv2.VideoWriter(f"{sav_dir}{Path(video_path).stem}/{video_name}_{i}.mp4", fourcc, fps, (w,h)) for i in range(1, out_frame_num+1)]

        f = 0
        
        # frame을 받아올 수 있을 때까지
        while ret:
            f += 1
    
            for i, start in enumerate(range(1, frame_num, step)):
                end = start+step
                if start <= f < end:
                    writers[i].write(frame)
            # next frame
            ret, frame = vc.read()
        # 자원 해제
        for writer in writers:
            writer.release()

        vc.release()
        print('split done.')
