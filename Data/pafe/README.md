# PAFE Dataset

## Description

Each folder in `./data` directory contains each participant's data of:

1. Facial video (`recording.mp4`)
1. Corresponding frame timeline (`video_timeline.txt`)
1. Program execution log (`main_log.txt`)
1. Probe response log (`probe_main_video.txt`)

## How to use

OpenCV does not load the consecutive duplicated (i.e. repeated) frames from the video. Please use `ffmpeg` (or equivalent) to extract raw frames from the facial video.

- You can use `./extract_frames.py` with `numpy` package installed: `$ python3 extract_frames.py`

## Citation
```Taeckyung Lee, Dain Kim, Sooyoung Park, Dongwhi Kim, Sung-Ju Lee; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022, pp. 2104-2113```

## Contact
taeckyung (at) kaist.ac.kr
