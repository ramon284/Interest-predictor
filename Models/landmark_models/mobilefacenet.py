import face_alignment
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torch

class landmarkDetector():
    def __init__(self, filename, df):
        self.filename = filename
        self.df = df
        self.video = cv2.VideoCapture('./Data/Video/'+filename)
        self.faceData = pd.DataFrame(columns=['Frame', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight'])
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=False)
        
        
    def runModel(self, batch_size = 8):
        current_batch = 0
        columns = ['Frame', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']
        for i in range(68):
            columns.extend([f'x{i}', f'y{i}'])
        result_df = pd.DataFrame(columns=columns)
        while True:
            image_batch = []
            for i in range(batch_size):
                ret, img = self.video.read()
                if ret == False:
                    break
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                image_batch.append(img)

            if len(image_batch) == 0:
                break

            frame_range = range(current_batch, current_batch + batch_size)
            filtered_df = self.df[self.df['Frame'].isin(frame_range)]

            mask = self.df['Frame'].isin(frame_range)
            groups = self.df.loc[mask].groupby('Frame')

            nested_list = []
            for frame, group_df in groups:
                sublist = group_df.loc[:, ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']].values.tolist()
                nested_list.append(sublist)

            nested_list = self.df.loc[frame_range, ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']].apply(lambda x: x.values.tolist(), axis=1).tolist()
            bounding_boxes_batch = [[(x, y, x+w, y+h) for x, y, w, h in (bboxes if isinstance(bboxes[0], list) else [bboxes])] for bboxes in nested_list]
            ## the rest of face detectors, mainly ``list[(x1,y1,x2,y2),...]``.

            image_batch = torch.stack(image_batch).to('cuda')
            ####detected_faces_batch = self.model.face_detector.detect_from_batch(image_batch)
            landmarks_batch = self.model.get_landmarks_from_batch(image_batch, detected_faces=bounding_boxes_batch)

            # Process the results
            for frame, landmarks_list in zip(frame_range, landmarks_batch):
                if landmarks_list is not None:
                    for landmarks in landmarks_list:
                        landmark_data = {}
                        for i, point in enumerate(landmarks):
                            x, y = point
                            landmark_data[f'x{i}'] = x
                            landmark_data[f'y{i}'] = y

                        # Update the DataFrame with the landmark data
                        mask = self.df['Frame'] == frame
                        row = self.df.loc[mask].iloc[0].to_dict()
                        row.update(landmark_data)
                        result_df = result_df.append(row, ignore_index=True)

            current_batch += batch_size
            torch.cuda.empty_cache()