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

def cook_video_analysis_multi_dir(module, video_dir, video_format, fps, db, font_dict, device, fourcc='mp4v', sav=False, split=False, food_score=90, font_text_score=0.95,
                                  text_match_score=0.9, freq=0, find_range=3, result_dir=None, seg_dir=None):
    
    """
    If there are multi directories that contains videoes, It can be used.
    """
    
    # filter dir_list by video format
    dir_list = os.listdir(video_dir)
    
    print(f"total_dir: {dir_list}")
    
    video_dir_tags = {}
    
    for playlist in font_dict.keys():
        
        try:
            if playlist in dir_list: #exist check

                print(f"playlist: {playlist} is running")

                new_video_dir = f"{video_dir}{playlist}/"

                new_dir_list = os.listdir(new_video_dir)
                new_dir_list = [d for d in new_dir_list if video_format in d]
                new_dir_tags = []

                for d in new_dir_list:

                    video_path = new_video_dir + d
                    video_name = Path(d).stem
                    if sav:
                        result_path = result_dir + video_name + '.json'

                    print(f"video: {video_name} is running")

                    fh_low, fh_high = font_dict[playlist]

                    video_tags = cook_video_analysis(module, video_path, fps, db, device, fh_low=fh_low, fh_high=fh_high)
                    print(video_tags)
                    new_dir_tags.append(video_tags)

                video_dir_tags[playlist] = new_dir_tags
        
        except:
            print(f'No match {playlist} of font_dict with video_dir')
            
    return video_dir_tags

#Test

#### hyper-parameter ####

# module = {'food':10000, 'scenetxt': 12000}
# video_dir = '/home/Cook_Video_Analysis/Dataset/test/'
# video_format = 'mp4'
# result_dir = '/home/Cook_Video_Analysis/Dataset/engine_result/'
# fps=1
# db = ingredient_df
# font_dict ={'cooking_log':(125,140), 'bar':(145,154), 'home':(120, 130)}
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# split=False 
# sav=False 
# font_text_score = 0.95 


#### dir-ver ####
# video_dir_tags = cook_video_analysis_multi_dir(module, video_dir, video_format, result_dir, fps, db, font_dict, device, \
#                                                sav=sav, split=split, font_text_score=font_text_score)