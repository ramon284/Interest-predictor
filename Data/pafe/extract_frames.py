import numpy as np
import subprocess
import json
import os

RAW_DATA_DIR = ".\data"
FRAME_FOLDER_NAME = "frames"


subprocesses = []
for dirname in os.listdir(RAW_DATA_DIR):
    print("-"*45)
    print(dirname)
    curr_dir = os.path.join(RAW_DATA_DIR, dirname)
    video_path = os.path.join(curr_dir, 'recording.mp4')
    frame_dir = os.path.join(curr_dir, FRAME_FOLDER_NAME)
    print(frame_dir)
    print(video_path)

    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)
        print('done')
    else:
        print("Frames for %s already exists. If you want to re-extract for this folder, please remove the following folder: %s" % (dirname, frame_dir))
        continue

    with open(os.path.join(curr_dir, 'video_timeline.txt'), 'r') as timeline:
        print("Total frames in timeline: ", len(timeline.readlines()))
    
    result = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    try:
        result = json.loads(result.stdout)['streams'][0]
    except KeyError:
        print("Error when loading video file.")
        continue
    
    frames = [f for f in os.listdir(frame_dir) if f.endswith('.png') and os.path.isfile(os.path.join(frame_dir, f))]
    if len(frames) != result['nb_frames']:
        proc = subprocess.Popen(['ffmpeg', '-i', video_path, '-vsync', '0', os.path.join(frame_dir, 'frame%06d.png')],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
        subprocesses.append(proc)

    print("-"*45)
    
for proc in subprocesses:
    proc.wait()

print("Frame Extracted.")
