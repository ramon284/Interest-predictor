import numpy as np
import pandas as pd
import cv2
from feat.facepose_detectors.img2pose.img2pose_test import Img2Pose
from torchvision import transforms

class face_detector:
    def __init__(self, filename,  detection_threshold=0.25, constrained=1):
        self.video = cv2.VideoCapture('./Data/Video/'+filename)
        self.filename = filename
        self.frame_i = 0
        self.faceData = pd.DataFrame(columns=['Frame', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 
                                              'FaceRectHeight', 'pose1', 'pose2', 'pose3'])
        self.model = Img2Pose(constrained=constrained, detection_threshold=detection_threshold, 
                              rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=200)

        
    def runModel(self):
        convert_tensor = transforms.ToTensor()
        frame_i = 0
        while True:
            ret, img = self.video.read()
            if ret == False:
                break
            img = convert_tensor(img)
            pred = self.model.scale_and_predict(img)
            for i in range(len(pred['boxes'])):
                top = pred['boxes'][i][1]
                right = pred['boxes'][i][2]
                bottom = pred['boxes'][i][3]
                left = pred['boxes'][i][0]
                pose1 = pred['poses'][i][0]
                pose2 = pred['poses'][i][1]
                pose3 = pred['poses'][i][2]
                self.faceData = pd.concat([self.faceData, 
                            pd.DataFrame({'Frame': frame_i,  'FaceRectX': left, 'FaceRectY': top, 
                                        'FaceRectWidth': right - left, 'FaceRectHeight': bottom - top,
                                        'pose1': pose1, 'pose2': pose2, 'pose3': pose3}, 
                                        index=[0])], 
                            ignore_index=True)
            frame_i += 1
        return self.faceData