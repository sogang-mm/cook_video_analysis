# -*- coding: utf-8 -*-
import copy
from collections import Counter
from utils.time_frame import frame_to_time, time_to_frame

def font_height_to_segmentation(data, video_name, score, fh_low, fh_high, freq, fps, find_range):
    
    """
    data: type = .json, help = scene_text_recognition result, e.g) result_30
    video_name: type = str, help = video name without the file extension like .mp4, e.g) 'bundaegi', 'egg', 'nuddle'
    score : type=float, help= scentxt_score, e.g) 0.7
    fh_low : type=int, help= font_height_low, e.g) 125
    fh_high : type=int, help= font_height_high, e.g) 140
    freq : type=int, help= frequency of text_bboxes per frame, max_value == fps, e.g) 10
                        (The larger the number, the higher the probability that the subtitle appeared.)
    fps : type=int, help=frame per second, e.g) 30
    find_range: type=int, help=sec, take text_result for time of find_range from segment start time, e.g) 3
    """
    
    frame_results_list = [] #frame 별 result만 따로 정리하자

    for idx in range(len(data[video_name]['scenetxt']['frame_results'])): #영상 전체 프레임 수 만큼 반복

        frame_results_list.append(data[video_name]['scenetxt']['frame_results'][idx]['frame_result'])
    
    target_frame_label = {} #filtered {frame : description(label)}

    for frame, result in enumerate(frame_results_list):
        for i in result:
            
            #scene_text_score& font height filter
            if (i['label'][0]['score'] > score) and (fh_low<=i['position']['h']<fh_high):
                target_frame_label[frame] = i['label'][0]['description']
    
    # frame to time& sort by asceding order
    target_time = sorted([frame_to_time(frame=j, fps=fps) for j in list(set(target_frame_label.keys()))], key= lambda x: (x[0], x[1]) )
    
    # frequency the font appeared filter: 1 프레임에서도 같은 높이의 폰트가 여러 번 등장할 수 있다. 즉 등장 빈도수를 freq에 대해 filtering
    cnt = Counter(target_time)
    most_freq_list = list(filter(lambda x: x[1] > freq, cnt.most_common()))
    
    most_list = sorted([i[0] for i in most_freq_list], key = lambda x: (x[0], x[1]))
    
    # Frames appearing in succession are represented by the front frame: 연달아 나오는 프레임은 맨 앞 시간으로 태그를 걸어주면 된다.
    seg_time = copy.deepcopy(most_list)

    for i in range(len(most_list)-1):

        if most_list[i+1][1] == most_list[i][1] + 1:
            seg_time.remove(most_list[i+1])
        else:
            pass
    
    # time to frame
    seg_frame = list(map(lambda x: time_to_frame(x,fps), seg_time))
    
    # take tags from seg_time for find_range time
    seg_tag = {}
    for start in seg_frame:
        tag_list = []
        end = start + (find_range * fps)

        for frame in target_frame_label.keys():
            if start <= frame < end:
                tag_list.append(target_frame_label[frame])
#         seg_tag[start] = set(tag_list) #frame: tag_list
        seg_tag[frame_to_time(start, fps)] = set(tag_list) #time: tag_list
    
    return seg_time, seg_tag
