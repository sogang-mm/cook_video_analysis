
def frame_to_time(frame, fps):
    """
    1. frame: int, frame number, e.g) 2958
    2. fps: int, frame per second, e.g) 30
    """
    sec = frame/fps
    minute = int(sec // 60)
    min_sec = sec % 60
    return (minute, int(min_sec))

def time_to_frame(time, fps):
    """
    1. time: tuple, (miniute, second), e.g) 1분 38초 = (1,38)
    2. fps: int, frame per second, e.g) 30
    """
    sec = time[0]*60 + time[1]
    frame = sec * fps
    return frame
