import numpy as np
import cv2
from feat import Detector
import pandas as pd
import os
import cv2

class AuEmoDetectors():
    def __init__(self, filename, df, emoModel='svm', device='cuda'):
        self.emoModel = emoModel
        self.filename = filename
        self.df = df
        self.video = cv2.VideoCapture('./Data/Video/'+filename)
        print('Video loaded? : ', self.video.isOpened())
        self.x_columns = [f'x{i}' for i in range(68)]
        self.y_columns = [f'y{i}' for i in range(68)]
        self.face_columns = ['FaceRectWidth', 'FaceRectHeight']
        self.bbox_columns = ['FaceRectX', 'FaceRectWidth', 'FaceRectY', 'FaceRectHeight']
        self.au_keys =      ["AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU11","AU12","AU14",
                             "AU15","AU17","AU20","AU23","AU24","AU25","AU26","AU28","AU43"]
        self.emotion_keys = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
        self.auDF = pd.DataFrame(index=range(len(df)), columns=self.au_keys)
        self.emoDF = pd.DataFrame(index=range(len(df)), columns=self.emotion_keys)
        self.detectorModel = Detector(emotion_model=emoModel, device=device)
        self.tempdf = ''
        

    def detectFeatures(self):
        frame_i = 0
        total_iterator = 0
        while True:
            ret, image = self.video.read()
            if ret == False:
                break
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            grouped = self.df.groupby('Frame')
            for frame, group in grouped:
                if frame == frame_i:
                    frame_group = group
                    break
            for _, row in frame_group.iterrows():
                x_array = row[self.x_columns].tolist()
                y_array = row[self.y_columns].tolist()
                coordinates = list(zip(x_array, y_array))
                coordinates = np.array(coordinates, dtype=np.float32)
                nested_coordinates = [[coordinates]]
                aus = self.detectorModel.detect_aus(image, nested_coordinates)
                if self.emoModel == 'svm':
                    emotions = self.detectorModel.detect_emotions(frame=image, facebox=['a','b'], landmarks=nested_coordinates)
                else:
                    faceInfo = row[self.bbox_columns].tolist()
                    faceInfo.append(1)
                    faceInfo = [int(x) for x in faceInfo]
                    faceInfo = [[faceInfo]]
                    emotions = self.detectorModel.detect_emotions(frame=image, facebox=faceInfo, landmarks=nested_coordinates)
                for name, key in zip(self.au_keys,aus[0][0]):
                    self.auDF.loc[total_iterator, name] = key
                for name, key in zip(self.emotion_keys, emotions[0][0]):
                    self.emoDF.loc[total_iterator, name] = key
                total_iterator += 1
    
        self.tempdf = self.df.copy(deep=True)
        self.tempdf = pd.concat([self.tempdf, self.auDF, self.emoDF], axis=1)
        self.tempdf.to_csv('final_with_emo.csv', index=False)

    def runModel(self):
        self.detectFeatures()
        return self.tempdf
        