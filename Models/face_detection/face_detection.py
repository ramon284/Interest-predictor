import numpy as np
import pandas as pd
import cv2
from feat.facepose_detectors.img2pose.img2pose_test import Img2Pose
from torchvision import transforms
from feat.utils.image_operations import convert_image_to_tensor
import torch
from concurrent.futures import ThreadPoolExecutor


class face_detector:
    def __init__(self, filename,  detection_threshold=0.5, constrained=1, batch_size = 8):
        self.video = cv2.VideoCapture('./Data/Video/'+filename)
        self.filename = filename
        self.frame_i = 0
        self.faceData = pd.DataFrame(columns=['Frame', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 
                                              'FaceRectHeight', 'Pitch', 'Roll', 'Yaw'])
        
        print(int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = Img2Pose(device = self.device, constrained=constrained, detection_threshold=detection_threshold)
        
    def runModel(self):
        frame_i = 0
        while True:
            ret, img = self.video.read()
            if ret == False:
                break
            img = convert_image_to_tensor(img, img_type="float32")[0] /255
            pred = self.model.scale_and_predict(img)
            if len(pred['boxes']) == 0:
                self.faceData = pd.concat([self.faceData,
                                pd.DataFrame({'Frame': frame_i, 'FaceRectX': np.nan, 'FaceRectY': np.nan,
                                            'FaceRectWidth': np.nan, 'FaceRectHeight': np.nan,
                                            'Pitch': np.nan, 'Roll': np.nan, 'Yaw': np.nan},
                                            index=[0])],
                                ignore_index=True)
            else:
                for i in range(len(pred['boxes'])):
                    top = pred['boxes'][i][1]
                    right = pred['boxes'][i][2]
                    bottom = pred['boxes'][i][3]
                    left = pred['boxes'][i][0]
                    Pitch = pred['poses'][i][0]
                    Roll = pred['poses'][i][1]
                    Yaw = pred['poses'][i][2]
                    if(left < 0 or right > self.frame_width or top < 0 or bottom > self.frame_height):
                        continue
                    self.faceData = pd.concat([self.faceData, 
                                pd.DataFrame({'Frame': frame_i,  'FaceRectX': left, 'FaceRectY': top, 
                                            'FaceRectWidth': right - left, 'FaceRectHeight': bottom - top,
                                            'Pitch': Pitch, 'Roll': Roll, 'Yaw': Yaw}, 
                                            index=[0])], 
                                ignore_index=True)
            frame_i += 1
        return self.faceData
    
    
    def runModelBatch(self, batch=True):
        frame_i = 0
        batch_frames = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                ret, img = self.video.read()
                if ret:
                    batch_frames.append((frame_i, img))
                    frame_i += 1
                if len(batch_frames) == self.batch_size or not ret:
                    if len(batch_frames) > 0:
                        
                        batch_frames_processed = list(executor.map(self.parallel_preprocess, batch_frames))
                        batch_imgs = [img for _, img in batch_frames_processed]
                        preds = self.model(batch_imgs, batch=batch)

                        for (frame_num, _), (boxes, poses) in zip(batch_frames, zip(preds[0], preds[1])):
                            if len(boxes) == 0:
                                self.faceData = pd.concat([self.faceData,
                                                pd.DataFrame({'Frame': frame_num, 'FaceRectX': np.nan, 'FaceRectY': np.nan,
                                                            'FaceRectWidth': np.nan, 'FaceRectHeight': np.nan,
                                                            'Pitch': np.nan, 'Roll': np.nan, 'Yaw': np.nan},
                                                            index=[0])],
                                                ignore_index=True)
                            else:
                                for i in range(len(boxes)):
                                    top = boxes[i][1]
                                    right = boxes[i][2]
                                    bottom = boxes[i][3]
                                    left = boxes[i][0]
                                    Pitch = poses[i][0]
                                    Roll = poses[i][1]
                                    Yaw = poses[i][2]
                                    if(left < 0 or right > self.frame_width or top < 0 or bottom > self.frame_height):
                                        continue
                                    self.faceData = pd.concat([self.faceData, 
                                                pd.DataFrame({'Frame': frame_num,  'FaceRectX': left, 'FaceRectY': top, 
                                                            'FaceRectWidth': right - left, 'FaceRectHeight': bottom - top,
                                                            'Pitch': round(Pitch, 3), 'Roll': round(Roll, 3), 'Yaw': round(Yaw, 3)}, 
                                                            index=[0])], 
                                                ignore_index=True)
                    batch_frames = []
                if not ret:
                    break

        return self.faceData
    
    
    def runModelBatchSingleFace(self, batch=True):
        frame_i = 0
        prev_bbox = None
        batch_frames = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                ret, img = self.video.read()
                if ret:
                    batch_frames.append((frame_i, img))
                    frame_i += 1
                if len(batch_frames) == self.batch_size or not ret:
                    if len(batch_frames) > 0:
                        
                        batch_frames_processed = list(executor.map(self.parallel_preprocess, batch_frames))
                        batch_imgs = [img for _, img in batch_frames_processed]
                        preds = self.model(batch_imgs, batch=batch)
                        
                        #batch_imgs = [img.to(self.device) for _, img in batch_frames]
                        #batch_imgs = [img for _, img in batch_frames]
                        #preds = self.model(batch_imgs)
                        for (frame_num, _), face_boxes, face_poses in zip(batch_frames, *preds):
                            if len(face_boxes) == 0:
                                self.faceData = pd.concat([self.faceData,
                                                pd.DataFrame({'Frame': frame_num, 'FaceRectX': np.nan, 'FaceRectY': np.nan,
                                                            'FaceRectWidth': np.nan, 'FaceRectHeight': np.nan,
                                                            'Pitch': np.nan, 'Roll': np.nan, 'Yaw': np.nan},
                                                            index=[0])],
                                                ignore_index=True)
                                prev_bbox = None
                            else:
                                if prev_bbox is None or len(face_boxes) == 1:
                                    idx = 0
                                else:
                                    distances = [self.distance(prev_bbox, bbox[:4]) for bbox in face_boxes]
                                    idx = distances.index(min(distances))
                                    
                                bbox = face_boxes[idx]
                                pose = face_poses[idx]
                                prev_bbox = bbox
                                    
                                if(bbox[0] < 0 or bbox[2] > self.frame_width or bbox[1] < 0 or bbox[3] > self.frame_height):
                                    continue
                                self.faceData = pd.concat([self.faceData, 
                                            pd.DataFrame({'Frame': frame_num,  'FaceRectX': bbox[0], 'FaceRectY': bbox[1], 
                                                        'FaceRectWidth': bbox[2] - bbox[0], 'FaceRectHeight': bbox[3] - bbox[1],
                                                        'Pitch': pose[0], 'Roll': pose[1], 'Yaw': pose[2]}, 
                                                        index=[0])], 
                                            ignore_index=True)
                    batch_frames = []
                if not ret:
                    break

        return self.faceData    
    
    
    def parallel_preprocess(self, frame_data):
        frame_i, img = frame_data
        img = convert_image_to_tensor(img, img_type="float32")[0] / 255
        return frame_i, img.to(self.device)
    
    def distance(self, rect1, rect2):
        x1, y1, w1, h1, _ = rect1
        x2, y2, w2, h2 = rect2
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5