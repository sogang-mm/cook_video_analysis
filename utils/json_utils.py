# -*- coding: utf-8 -*-

import json
from pathlib import Path

def save_load_json(file_path, sav = True, data=None):
    
    """
    1. sav: bool, default: True, 저장할 때 True, 불러올 때 False, e.g) sav=True(save)/ sav=False(load)
    2. file_path: str, 저장시 저장 파일 경로, 불러올 시 불러오는 파일 경로, e.g) '/Cook_Video_Analysis/Dataset/engine_result/meat.json'
    3. data: list, default=None, 저장할(if sav=True) 데이터, e.g) result (= analysis_engine(module. video_dir, video_format))
    """
    
    video_name = Path(file_path).stem
    
    if sav:
        with open(file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False) # indent: 들여쓰기(가독성)/ ensure_ascii=False : 아스키 코드 -> 유니코드
            
        print(f"{video_name} save done.")
    else: #load
        with open(file_path, "r") as json_file:
            json_data = json.load(json_file)
        print(f"{video_name} load done.")
        return json_data
