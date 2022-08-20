# SetUp
import os
from pathlib import Path
import pandas as pd
import torch
from mmaction.apis import init_recognizer, inference_recognizer
from model.run_model import communicator
from model.run_analysis_engine import analysis_engine
from utils.json_utils import save_load_json
from utils.preprocess import preprocess
from utils.splitter import splitter
from utils.time_frame import frame_to_time, time_to_frame
from text_simple_match import text_simple_match
from font_height_to_segmentation import font_height_to_segmentation
from action_recognition import action_recognition

# Load Ingredient DB
ingredient_df = pd.read_csv('./Dataset/db/refined/ingredients.csv', encoding = 'cp949' )

def cook_video_analysis_dir(module, video_dir, video_format, fps, db, device, fourcc='mp4v', sav=False, split=False, food_score=90, font_text_score=0.95, text_match_score=0.9, 
                            fh_low=0, fh_high=200, freq=0, find_range=3, result_dir=None, seg_dir==None):
    """
    If there are videoes in a directory, It can be used.
    """
    
    # filter dir_list by video format
    dir_list = os.listdir(video_dir)
    dir_list = [d for d in dir_list if video_format in d]
    
    video_dir_tags = []
    for d in dir_list:
        video_path = video_dir + d
        if sav:
            result_path = result_dir + Path(d).stem + '.json'#if sav=True일 때만
        
        video_tags = cook_video_analysis(module, video_path, fps, db, device)
        
        video_dir_tags.append(video_tags)
    
    return video_dir_tags

#Test

#### hyper-parameter ####

# module = {'food':10000, 'scenetxt': 12000}
# video_dir = '/home/Cook_Video_Analysis/Dataset/test/'
# result_dir = '/home/Cook_Video_Analysis/Dataset/engine_result/'
# video_format = 'mp4'
# fps=1
# db = ingredient_df
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# split=False 
# sav=False 
# font_text_score = 0.95

#### dir-ver ####
# video_dir_tags = cook_video_analysis_dir(module, video_dir, video_format, result_dir, fps, db, device, sav=sav, split=split, font_text_score=font_text_score)