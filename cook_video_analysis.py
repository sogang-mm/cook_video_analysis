# -*- coding: utf-8 -*-
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

def cook_video_analysis(module, video_path, fps, db, device, fourcc='mp4v', sav=False, split=False, food_score=90, font_text_score=0.95, text_match_score=0.9, 
                        fh_low=0, fh_high=200, freq=0, find_range=3, result_path=None, seg_dir=None):
    
    """
    1. module: dictionary, 모듈 이름과 엔진 주소 번호, e.g) {'asr':9002, 'food':10000, 'places':10001, 'obj': 11000, 'scenetxt': 12000}
    2. video_path: string,  비디오 영상의 저장되어 있는 파일 주소, e.g) 'C:/Users/jcjo/Desktop/code/NotGit/Cook_Video_Analysis/Dataset/meat.mp4'
    3. fps: int, frame per second, e.g) 30
    4. db: pd.Dataframe, 매칭할 데이터 베이스 e.g) 식재료 데이터 베이스, df (= pd.read_csv('./Dataset/dataset/refined/ingredients.csv', encoding = 'cp949' ))
    5. device: str, 사용할 device e.g) 'cuda:0' or 'cpu'
    
    ### default exist ###
    6. fourcc: str, default='mp4v', four character code : 코덱, 압축 방식, 색상, 픽셀 포맷 등을 정의하는 정수 값 e.g) 'mp4v'
    7. sav: bool, defaul=False, 저장할 때 True, 불러올 때 False, e.g) sav=True(save)/ sav=False(load)
    8. split: bool, default=False, 영상 분할을 할 것인지, False시 통 영상 채로 분석 결과 도출 e.g) False
    9. food_score : int(0-100), default=90, food_classification_result 필터링할 기준 점수 값 e.g) 90
    10. font_text_score : int(0-1), default=0.95, font_height_to_segment할 때 scenetext_result 필터 기준 점수 값  e.g) 0.95
    11. text_match_score : int(0-1), default=0.9, text_simple_match할 때 text_query 필터 기준 점수 값  e.g) 0.9
    12. fh_low : type=int, default=0, help= font_height_low(폰트 탐색 범위 하한), e.g) 125
    13. fh_high : type=int, default=200, help= font_height_high(폰트 탐색 범위 상한), e.g) 140
    14. freq : type=int, default=0, help= frequency of text_bboxes per frame(프레임 당 텍스트 박스 출현 빈도 수 필터 기준 개수), e.g) 10
               (*freq <= fps) (The larger the number, the higher the probability that the subtitle appeared.)
    15. find_range: type=int, default=3, help=sec, take text_result for time of find_range from segment start time(구간 기준 시간으로부터 몇 초 태그 리스트 탐색 범위), e.g) 3 
    16. result_path: str, default=None, 저장시 저장 파일 경로, 불러올 시 불러오는 파일 경로, e.g) '/Cook_Video_Analysis/Dataset/engine_result/meat.json'
    17. seg_dir: str, default=None, 구간 분할한 영상 저장할 경로 e.g) '/home/Cook_Video_Analysis/Dataset/segment/'
    """
    
    # engine-server communicator instance
    com = communicator()
    
    #analysis by module engines
    result = analysis_engine(module, video_path, fps)
    #result = result##test###
    
    #video_name
    video_name = Path(video_path).stem
    
    #if you want to save
    if sav:
        save_load_json(file_path=result_path, sav=sav, data=result)
    
    #tag_list about video -> pd.Dataframe
    video_tags = {}
    
    ####### Vision Analysis #######
    
    #food_classification
    food = preprocess(data=result, score=food_score, video_name=video_name, module_name='food')
    #top1_tag
    video_tags['food'] = pd.Series(food).value_counts().index[0]
    print('food_cls done.')
    
    #Action_recognition
    config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    checkpoints_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    top3_tags = action_recognition(video_path=video_path, sav_dir=seg_dir, config_file=config_file, checkpoint_file=checkpoints_file, \
                                   device=device, topk=3, split=split, segment_sec=10, fourcc=fourcc)
    #top3_tags
    video_tags['action_recog'] = top3_tags
    print('action_recog done.')
    
    ####### Text Analysis ########
    
    #Segmentation of video
    _, seg_tag = font_height_to_segmentation(data=result, video_name=video_name, score=font_text_score, fh_low=fh_low, fh_high=fh_high, \
                                             freq=freq, fps=fps, find_range=find_range)
    #frame:tag
    video_tags['segment'] = seg_tag
    print('segment done.')
    
    #text_simple_match
    text = preprocess(data=result, score=text_match_score, video_name=video_name, module_name='scenetxt')
    #top5_tags
    top5_tags = text_simple_match(data=text, db=db, topk=5)
    
    video_tags['match'] = top5_tags
    print('match done.')
    
    #video_name
    video_tags['video'] = video_name
    
    return video_tags

# +
# Test

# module = {'food':10000, 'scenetxt': 12000} 
# video_path = '/home/Cook_Video_Analysis/Dataset/test/mooksabal.mp4'
# fps=30
# db = ingredient_df
# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# split=True
# sav=False 
# seg_dir = '/home/Cook_Video_Analysis/Dataset/segment/'
# font_text_score = 0.95 
# fh_low=87
# fh_high=93

# video_tags = cook_video_analysis(module, video_path, fps, db, device, sav=sav, split=split, seg_dir=seg_dir, \
#                                  font_text_score=font_text_score, fh_low=fh_low, fh_high=fh_high)
