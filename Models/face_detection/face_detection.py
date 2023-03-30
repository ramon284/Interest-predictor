import numpy as np
import pandas as pd
import cv2
from feat.facepose_detectors.img2pose.img2pose_test import Img2Pose
from torchvision import transforms
from feat.utils.image_operations import convert_image_to_tensor

class face_detector:
    def __init__(self, filename,  detection_threshold=0.5, constrained=1):
        self.video = cv2.VideoCapture('./Data/Video/'+filename)
        self.filename = filename
        self.frame_i = 0
        self.faceData = pd.DataFrame(columns=['Frame', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 
                                              'FaceRectHeight', 'Pitch', 'Roll', 'Yaw'])
        self.model = Img2Pose(constrained=constrained, detection_threshold=detection_threshold)
        
    def runModel(self):
        ##convert_tensor = convert_image_to_tensor()
        frame_i = 0
        while True:
            ret, img = self.video.read()
            if ret == False:
                break
            img = convert_image_to_tensor(img, img_type="float32")[0] /255
            pred = self.model.scale_and_predict(img)
            for i in range(len(pred['boxes'])):
                top = pred['boxes'][i][1]
                right = pred['boxes'][i][2]
                bottom = pred['boxes'][i][3]
                left = pred['boxes'][i][0]
                Pitch = pred['poses'][i][0]
                Roll = pred['poses'][i][1]
                Yaw = pred['poses'][i][2]
                self.faceData = pd.concat([self.faceData, 
                            pd.DataFrame({'Frame': frame_i,  'FaceRectX': left, 'FaceRectY': top, 
                                        'FaceRectWidth': right - left, 'FaceRectHeight': bottom - top,
                                        'Pitch': Pitch, 'Roll': Roll, 'Yaw': Yaw}, 
                                        index=[0])], 
                            ignore_index=True)
            frame_i += 1
        return self.faceData