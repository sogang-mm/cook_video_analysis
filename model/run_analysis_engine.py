# -*- coding: utf-8 -*-
# %%
from pathlib import Path
from model.run_model import communicator


# %%
def analysis_engine(module, video_path, fps):
    
    """
    1. video_path: string,  비디오 영상의 저장되어 있는 파일 주소, e.g) 'C:/Users/jcjo/Desktop/code/NotGit/Cook_Video_Analysis/Dataset/meat.mp4'
    2. module: dictionary, 모듈 이름과 엔진 주소 번호, e.g) {'asr':9002, 'food':10000, 'places':10001, 'obj': 11000, 'scenetxt': 12000}
    3. video_format: list, 비디오 확장자 리스트, e.g) ['mp4', 'avi']
    4. fps: int, frame per second, e.g) 30
    """
    
    # engine-server communicator instance
    com = communicator()
    
    # video_name
    video_name = Path(video_path).stem
    
    result={}
    
    result_category = {}
    
    for name, addr in module.items():

        print(f"{name}-engine is running") 

        if name == 'asr': #asr(audio-speech-recognition) model cannot use temporarily
            pass
        else:
            result_video = com.communicator_video(f"http://mllime.sogang.ac.kr:{addr}/video/", video_path, "", fps, "", "", "video") #30 : 30fps
            result_category[name] = result_video
        
    result[video_name] = result_category
    
    return result
