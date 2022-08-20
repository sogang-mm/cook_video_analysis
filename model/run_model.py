# hidf-engine-asr_main : Automatic Speech Recognition
# hidf-engine-food_main : Food Detection(yolov4 + resnet50)
# hidf-engine-places_main : Places Recognition(resnet)
# hidf-engine-object_main : Object Detection(efficientdet)
# hidf-engine-scenetext_main : Scene text recognition
#
# hidf-engine-asr_main               http://mlamethyst.sogang.ac.kr:9002
# hidf-engine-food_main              http://mlamethyst.sogang.ac.kr:10000
# hidf-engine-places_main            http://mlamethyst.sogang.ac.kr:10001
# hidf-engine-object_main            http://mlamethyst.sogang.ac.kr:11000
# hidf-engine-scenetext_main         http://mlamethyst.sogang.ac.kr:12000

import os
import time
import requests
import json
from os import path
from PIL import Image

class communicator:

    def __init__(self):
        self.count = 0

    def communicator_image(self, module_url, image_path):

        json_data = dict()
        json_image = open(image_path, 'rb')
        json_files = {'image': json_image}

        result_response = requests.post(url= module_url, data=json_data, files=json_files)
        result_data = json.loads(result_response.content)
        result = result_data['result']

        json_image.close()

        return result

    def communicator_video(self, module_url, video_path, video_text, extract_fps, start_time, end_time, module_type):
        json_video = open(video_path, 'rb')
        json_files = {'video': json_video}
        json_data = dict({
            "analysis_type": module_type,
            "video_text": video_text,
            "extract_fps": extract_fps,
            "start_time": start_time,
            "end_time": end_time
        })
        result_response = requests.post(url=module_url, data=json_data, files=json_files) #data get sended, files get back
        result_data = json.loads(result_response.content)
        result = result_data['result']

        return result


if __name__ == '__main__':
    print("----------Test Start----------")
    com = communicator()
    result_img = com.communicator_image("http://mlamethyst.sogang.ac.kr:11000/image/", "Dataset/test/egg_Moment.jpg")
    # result = communicator_image("http://mlamethyst.sogang.ac.kr:11000/image/", "image/test.jpg")
    result_video = com.communicator_video("http://mlamethyst.sogang.ac.kr:10000/video/", "Dataset/egg.mp4", "", 1, "", "", "video")
    # result = communicator_video("http://mlamethyst.sogang.ac.kr:12000/video/", "media/video_name.mp4", "", 1,
    #                             "00:03:00", "00:04:00", "video")
    print(result_img)
    print(result_video)

    print("----------Test End----------")