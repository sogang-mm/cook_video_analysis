# -*- coding: utf-8 -*-

def preprocess(data, score, video_name, module_name):
    
    """
    1. data: json, analysis-engine 결과 data e.g) e.g) result (= analysis_engine(module. video_dir, video_format))
    2. score: int, result score 값으로 몇 점 이상으로 filter 할 것인지 e.g) 90(food: 0-100), 0.9(obj, scenetxt, places:0-1)
    3. video_name: str, 파일 확장자를 제외한 비디오 이름 e.g) egg.mp4 -> egg
    4. module_name: str, module.keys() 참고 e.g) module.keys():{'food', 'obj', 'scenetxt', 'places', 'asr'} -> module_name='food'
    """

    label_list = []

    for fr_results in data[video_name][module_name]['frame_results']:
        if (fr_results['frame_result'] != None):

            for i in range(0, len(fr_results['frame_result'])): #객체 수만큼 draw

                if fr_results['frame_result'][i]['label'][0]['score'] > score:

                    label = fr_results['frame_result'][i]['label'][0]['description']
                    label_list.append(label)
    return label_list
