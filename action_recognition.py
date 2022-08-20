# -*- coding: utf-8 -*-
import os
import torch
from pathlib import Path
from utils.splitter import splitter
from mmaction.apis import init_recognizer, inference_recognizer

def action_recognition(video_path, sav_dir, config_file, checkpoint_file, device, topk, split=False, segment_sec=None, fourcc=None):
    
    """
    1. video_path: str, action_recognition 돌릴 영상 주소 e.g) '/home/Cook_Video_Analysis/Dataset/meat.mp4'
    2. sav_dir: str, 구간 분할한 영상 저장할 경로 e.g) '/home/Cook_Video_Analysis/Dataset/segment/'
    3. config_file: str, action_recognition config(module) e.g) 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    4. checkpoints_file: str, weight check point 주소 e.g) 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    5. device: str, 사용할 device e.g) 'cuda:0' or 'cpu'
    6. topk: int, 몇 개의 빈도수 상위 태그를 뽑을 것인지 e.g) 3
    7. split: bool, default=False, 영상 분할을 할 것인지, False시 통 영상 채로 분석 결과 도출 e.g) False
    8. segment_sec: int, default=None, 몇 초씩 구간 분할을 할 것인지 e.g) 10
    9. fourcc: str, default=None, four character code : 코덱, 압축 방식, 색상, 픽셀 포맷 등을 정의하는 정수 값 e.g) 'mp4v'
    """
    
    # sort segment in ascending order
    video_name = Path(video_path).stem
    
    # assign the desired device.
    device = torch.device(device)
    
    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, checkpoint_file, device=device)
    
    # label open
    labels = open('tools/data/kinetics/label_map_k400.txt').readlines()
    labels = [x.strip() for x in labels]
    
    # split video into segments, and output
    if split:
        splitter(video_path, segment_sec, sav_dir, fourcc)

        seg_list = os.listdir(f'{sav_dir}{video_name}/')
        seg_sort = sorted(seg_list, key = lambda x: int(Path(x).stem[len(video_name)+1:]))
    
        results_dict = {}

        # action recognition execute with seg videoes
        for name in seg_sort:
            video = f'{sav_dir}{video_name}/{name}'
            results = inference_recognizer(model, video)

            # results
            results = [(labels[k[0]], k[1]) for k in results]
            results_dict[name] = results
        
        # analysis result by value count
        results_count = {}

        for key in results_dict.keys():

            for label, score in results_dict[key]:

                if label in results_count:
                    results_count[label] += score
                else:
                    results_count[label] = score
                    
        # topk label tag list
        topk_tags = [label[0] for label in sorted(results_count.items(), key = lambda x: -x[1])[:topk]]
        
    else: # not split video output
        results = inference_recognizer(model, video=video_path)
        topk_tags = [labels[k[0]] for k in results[:topk]]
        
    return topk_tags
