import numpy as np
import cv2
from feat import Detector
import pandas as pd
import os
import cv2
from feat.utils.image_operations import convert_image_to_tensor
import torch
from datetime import datetime

class AuEmoDetectors():
    def __init__(self, filename, df, emoModel='resmasknet', device='cuda', emo_cuda=True):
        self.emoModel = emoModel
        self.filename = filename
        self.df = df
        self.video = cv2.VideoCapture('./Data/Video/'+filename)
        print('Video loaded? : ', self.video.isOpened())
        self.x_columns = [f'x{i}' for i in range(68)]
        self.y_columns = [f'y{i}' for i in range(68)]
        self.bbox_columns = ['FaceRectX', 'FaceRectWidth', 'FaceRectY', 'FaceRectHeight']
        self.au_keys =      ["AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU11","AU12","AU14",
                             "AU15","AU17","AU20","AU23","AU24","AU25","AU26","AU28","AU43"]
        self.emotion_keys = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
        self.auDF = pd.DataFrame(index=range(len(df)), columns=self.au_keys)
        self.emoDF = pd.DataFrame(index=range(len(df)), columns=self.emotion_keys)
        self.detectorModel = Detector(emotion_model=emoModel, device=device, emo_cuda=emo_cuda)
        
    def re_init_video(self):
        del self.video
        self.video = cv2.VideoCapture('./Data/Video/'+self.filename)

    def runModel(self, batch_size=32, separate=False):
        if separate:
            self.detectAUOnlyCPU()
            self.re_init_video()
            self.df = self.tempdf
            self.detectEmotions()
            return self.tempdf
        self.detectBoth(batch_size = batch_size)
        return self.tempdf
    
    def detectEmotions(self, batch_size=32):
        frame_i = 0
        total_iterator = 0
        while True:
            batch_images = []
            final_faceBoxList = []
            for i in range(batch_size):
                ret, image = self.video.read()
                if ret == False:
                    break
                image = convert_image_to_tensor(image, img_type="float32")[0]

                grouped = self.df.groupby('Frame')
                for frame, group in grouped:
                    if frame == frame_i:
                        frame_group = group
                        break
                faceBoxList = []
                
                for _, row in frame_group.iterrows():
                    faceInfo = row[self.bbox_columns].tolist()
                    left, width, top, height = faceInfo
                    right = left + width
                    bottom = top + height
                    faceInfo = [left, top, right, bottom]
                    faceInfo.append(1)
                    faceInfo = [float(x) for x in faceInfo]
                    faceBoxList.append(faceInfo)

                final_faceBoxList.append(faceBoxList)
                batch_images.append(image)
                frame_i += 1
                
            if not batch_images:
                break

            emotions = self.detectorModel.detect_emotions_batch(frames=batch_images, faceboxes=final_faceBoxList, landmarks='x')

            for batch_emotions in emotions:
                for person in range(len(batch_emotions)):           
                    for name, key in zip(self.emotion_keys, batch_emotions[person]):
                        self.emoDF.loc[total_iterator, name] = key
                    total_iterator += 1

        self.tempdf = self.df.copy(deep=True)
        self.tempdf = pd.concat([self.tempdf, self.emoDF], axis=1)
        self.tempdf.to_csv(f'final_{self.filename}.csv', index=False)
        return self.tempdf

        
    def detectAUOnlyCPU(self, batch_size=32):
        frame_i = 0
        total_iterator = 0
        while True:
            # Read frames and their corresponding landmarks in a batch
            batch_images = []
            batch_landmarks = []
            for _ in range(batch_size):
                ret, image = self.video.read()
                if not ret:
                    break
                image = convert_image_to_tensor(image, img_type="float32")[0]
                grouped = self.df.groupby('Frame')
                for frame, group in grouped:
                    if frame == frame_i:
                        frame_group = group
                        break
                nested_coordinates = []

                for _, row in frame_group.iterrows():
                    x_array = row[self.x_columns].tolist()
                    y_array = row[self.y_columns].tolist()
                    coordinates = list(zip(x_array, y_array))
                    coordinates = np.array(coordinates, dtype=np.float32)
                    coordinates = [coordinates]
                    nested_coordinates.append(coordinates[0])

                batch_images.append(image)
                batch_landmarks.append(nested_coordinates)
                frame_i += 1

            if not batch_images:
                break
            # Call the detect_aus function with a batch of frames and landmarks
            batch_landmarks = np.array(batch_landmarks)
            aus = self.detectorModel.detect_aus_batch_cpu(batch_images, batch_landmarks)
            # Process the detected Action Units
            for au_batch in aus:
                for person in range(len(au_batch)):
                    for name, key in zip(self.au_keys, au_batch[person]):
                        self.auDF.loc[total_iterator, name] = key
                    total_iterator += 1

        self.tempdf = self.df.copy(deep=True)
        self.tempdf = pd.concat([self.tempdf, self.auDF], axis=1)
        self.tempdf.to_csv(f'final_{self.filename}.csv', index=False)
        return self.tempdf
    
    
    def detectBoth(self, batch_size=32):
        frame_i = 0
        total_iterator = 0
        while True:
            batch_images = []
            final_faceBoxList = []
            batch_landmarks = []
            for _ in range(batch_size):
                ret, image = self.video.read()
                if ret == False:
                    break
                image = convert_image_to_tensor(image, img_type="float32")[0]
                grouped = self.df.groupby('Frame')
                try:
                    frame_group = grouped.get_group(frame_i)
                except KeyError:
                    frame_group = pd.DataFrame()
     
                nested_coordinates = []
                faceBoxList = []
                                
                faceInfo_array = frame_group[self.bbox_columns].values
                right_bottom = faceInfo_array[:, [0, 2]] + faceInfo_array[:, [1, 3]]
                faceBoxList = np.hstack([faceInfo_array[:, [0, 2]], right_bottom, np.ones((faceInfo_array.shape[0], 1), dtype=np.float32)])

                x_array = frame_group[self.x_columns].values
                y_array = frame_group[self.y_columns].values
                coordinates = np.stack([x_array, y_array], axis=-1).astype(np.float32)
                nested_coordinates = [coord for coord in coordinates]
                
                final_faceBoxList.append(faceBoxList)
                
                batch_images.append(image)
                batch_landmarks.append(nested_coordinates)
                frame_i += 1
                
            if not batch_images:
                break
            
            batch_images_tensor = torch.stack(batch_images, dim=0)

            batch_landmarks = np.array(batch_landmarks)
            aus = self.detectorModel.detect_aus_batch_cpu(batch_images_tensor, batch_landmarks)
            emotions = self.detectorModel.detect_emotions_batch(frames=batch_images_tensor, faceboxes=final_faceBoxList, landmarks='x')
                    
            for batch_emotions, au_batch in zip(emotions, aus):
                for person in range(len(batch_emotions)):
                    for name, key in zip(self.au_keys, au_batch[person]):
                        self.auDF.loc[total_iterator, name] = round(key, 4)
                    for name, key in zip(self.emotion_keys, batch_emotions[person]):
                        self.emoDF.loc[total_iterator, name] = round(key, 4)
                        
                    total_iterator +=1

        self.tempdf = self.df.copy(deep=True)
        self.tempdf = pd.concat([self.tempdf, self.auDF, self.emoDF], axis=1)
        self.tempdf.to_csv(f'final_{self.filename}.csv', index=False)
        return self.tempdf