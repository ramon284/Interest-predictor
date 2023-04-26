## Experimenting with saving face_recognition face data with pickle
## Based on some experimenting with mtcnn and retinaface models, these methods may be preferred for general face detection

import face_recognition
import os
import cv2
import pickle
import time
import numpy as np
import pandas as pd
from feat import Detector
import torch
from feat.utils.image_operations import convert_image_to_tensor

class face_persistence_model:
    def __init__(self, TOLERANCE, filename, model='cnn', modelSize='large', landmark_cuda=True):
        self.KNOWN_FACES = "./known_faces"
        self.UNKNOWN_FACES = "./unknown_faces"
        self.TOLERANCE = TOLERANCE ## <- most important hyperparameter. Lower means it more quickly recognizes a "new" face
        self.FRAME_THICCNESS = 3
        self.FONT_THICCNESS = 2
        self.MODEL = model
        self.modelSize = modelSize
        self.filename = filename
        self.fileDir = self.KNOWN_FACES + '/' + self.filename[:-4]
        self.video = cv2.VideoCapture('./Data/Video/'+filename)
        #video = cv2.VideoCapture('Data/Video/Video0_Trim.mp4')
        print('Video loaded? : ', self.video.isOpened())
        self.faceData = []
        self.known_faces = []
        self.known_names = []
        self.next_id = 0
        self.faceData = pd.DataFrame(columns=['Frame','Person', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight'])
        self.frame_i = 0
        self.landmark_dict = {
            "chin": list(range(17)),
            "left_eyebrow": list(range(17, 22)),
            "right_eyebrow": list(range(22, 27)),
            "nose_bridge": list(range(27, 31)),
            "nose_tip": list(range(31, 36)),
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48)),
            "top_lip": list(range(48, 55)) + [64, 63, 62, 61, 60],
            "bottom_lip": list(range(54, 60)) + [48, 60, 67, 66, 65, 64]
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.myDetector = Detector(landmark_model='mobilefacenet', device=self.device, landmark_cuda = landmark_cuda)
        self.initDirectory()
        
    def initDirectory(self):    
        if not os.path.exists(self.fileDir):
            os.makedirs(self.fileDir)
        for name in os.listdir(self.fileDir):
            if name == '.gitignore':
                continue
            for filename in os.listdir(f"{self.fileDir}/{name}"):
                encoding = pickle.load(open(f"{self.fileDir}/{name}/{filename}", "rb"))
                self.known_faces.append(encoding)
                self.known_names.append(int(name))

        if len(self.known_names) > 0:
            self.next_id = max(self.known_names) + 1
        else:
            self.next_id = 0

    
    def runDetectionTrackOnly(self, someDF, display=False):
        someDF['Person'] = pd.Series([None] * len(someDF.index))
        empty_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
        someDF = pd.concat([someDF, empty_df], axis=1)
        frame_i = 0
        total_iterator = 0
        while True:
            ret, image = self.video.read()
            if ret == False:
                break
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) / 255
            frame_rows = someDF[someDF['Frame'] == self.frame_i]
            locations = [(int(row['FaceRectY']), int(row['FaceRectX'] + row['FaceRectWidth']), 
                          int(row['FaceRectHeight'] + row['FaceRectY']), int(row['FaceRectX'])) for _, row in frame_rows.iterrows()]
            locationsPyFeat = [(int(row['FaceRectX']), int(row['FaceRectY']), 
                          int(row['FaceRectX'] + row['FaceRectWidth']), int(row['FaceRectY'] + row['FaceRectHeight'])) for _, row in frame_rows.iterrows()]
            locationsPyFeatList = []
            for feats in locationsPyFeat:
                temp = [xyz for xyz in feats]
                temp.append(1)
                locationsPyFeatList.append(temp)
            locationsPyFeatList = [locationsPyFeatList]
            landmarks = self.myDetector.detect_landmarks(image, locationsPyFeatList)
            
            ## top, right, bottom, left
            encodings = face_recognition.face_encodings(image, locations, model=self.modelSize)
            #landmarks = face_recognition.face_landmarks(image, locations, model=self.modelSize)
            
            for face_encoding, face_location, face_landmark in zip(encodings, locationsPyFeat, landmarks[0]):
                #top, right, bottom, left = face_location
                left, top, right, bottom = face_location 
                results = face_recognition.compare_faces(self.known_faces, face_encoding, self.TOLERANCE)
                match = None
                if True in results:
                    match = self.known_names[results.index(True)]
                else:
                    match = str(self.next_id)
                    self.next_id += 1
                    self.known_names.append(match)
                    self.known_faces.append(face_encoding)
                    os.mkdir(f"{self.fileDir}\{match}")
                    pickle.dump(face_encoding, open(f"{self.fileDir}\{match}\{match}-{int(time.time())}.pkl", 'wb'))

                ## This is for visualizing the bounding boxes in real-time, with annotations
                if display == True:
                    top_left = (left, top)
                    bottom_right = (right, bottom)
                    color = [255, 255, 0]
                    cv2.rectangle(image, top_left, bottom_right, color, self.FRAME_THICCNESS)
                    top_left = (left, bottom)
                    bottom_right = (right, bottom+22)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, str(match), (left+10, bottom+15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0))
                
                loc = someDF.loc[(someDF['Frame'] == self.frame_i) & (someDF['FaceRectY'] == top) & (someDF['FaceRectX'] == left)].index[0]
                someDF.loc[loc, 'Person'] = str(match)
                
                temp_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
                
                for j, coords in enumerate(face_landmark):
                    cordX, cordY = coords
                    temp_df.at[0, f'x{j}'] = cordX
                    temp_df.at[0, f'y{j}'] = cordY
                        
                cols_to_update = temp_df.columns.intersection(someDF.columns)
                # Update only the required columns
                someDF.loc[loc, cols_to_update] = temp_df.loc[0, cols_to_update]
                
            if display == True:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imshow('x', image)
                if cv2.waitKey(1) & 0xFF ==ord('e'):
                    break
            self.frame_i += 1
            
        self.video.release()
        cv2.destroyAllWindows()
        return someDF
    
    def saveCSV(self, input=''):
        if isinstance(input, pd.DataFrame):
            input.to_csv('facial_persistence_test_'+self.filename+'.csv', index=False)
        else:
            self.faceData.to_csv('facial_persistence_test_'+self.filename+'.csv', index=False)
            
            
    def runModel_single_person(self, someDF, display=False):
        someDF['Person'] = pd.Series([None] * len(someDF.index))
        empty_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
        someDF = pd.concat([someDF, empty_df], axis=1)
        total_iterator = 0
        while True:
            ret, image = self.video.read()
            if ret == False:
                break
            
            frame_row = someDF[someDF['Frame'] == self.frame_i].iloc[0]
            location = (int(frame_row['FaceRectY']), int(frame_row['FaceRectX'] + frame_row['FaceRectWidth']), 
                        int(frame_row['FaceRectHeight'] + frame_row['FaceRectY']), int(frame_row['FaceRectX']))
            locationPyFeat = (int(frame_row['FaceRectX']), int(frame_row['FaceRectY']), 
                        int(frame_row['FaceRectX'] + frame_row['FaceRectWidth']), int(frame_row['FaceRectY'] + frame_row['FaceRectHeight']))
            locationsPyFeatList = [feats for feats in locationPyFeat]
            locationsPyFeatList.append(1)
            landmarks = self.myDetector.detect_landmarks(image, [[locationsPyFeatList]])
            
            match = 1
            loc = someDF.loc[(someDF['Frame'] == self.frame_i) & (someDF['FaceRectY'] == location[0]) & (someDF['FaceRectX'] == location[3])].index[0]
            someDF.loc[loc, 'Person'] = match
            
            temp_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
            
            for j, coords in enumerate(landmarks[0][0]):
                cordX, cordY = coords
                temp_df.at[0, f'x{j}'] = cordX
                temp_df.at[0, f'y{j}'] = cordY
                    
            cols_to_update = temp_df.columns.intersection(someDF.columns)
            # Update only the required columns
            someDF.loc[loc, cols_to_update] = temp_df.loc[0, cols_to_update]

            if display:
                left, top, right, bottom = location
                top_left = (left, top)
                bottom_right = (right, bottom)
                color = [255, 255, 0]
                cv2.rectangle(image, top_left, bottom_right, color, self.FRAME_THICCNESS)
                top_left = (left, bottom)
                bottom_right = (right, bottom+22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, str(match), (left+10, bottom+15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imshow('x', image)
                if cv2.waitKey(1) & 0xFF ==ord('e'):
                    break
            self.frame_i += 1
                
        self.video.release()
        if display:
            cv2.destroyAllWindows()
        return someDF         
            
    def runDetectionTrackOnlyBatch(self, someDF, display=False, batch_size=16):
        #someDF['Person'] = pd.Series([None] * len(someDF.index))
        empty_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
        someDF = pd.concat([someDF, empty_df], axis=1)
        frame_i = 0
        total_iterator = 0

        frames = []
        locations_batches = []

        while True:
            ret, image = self.video.read()
            if not ret:
                break
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) / 255
            image = convert_image_to_tensor(image, img_type="float32")[0]
            frame_rows = someDF[someDF['Frame'] == self.frame_i]

            locationsPyFeat = [(int(row['FaceRectX']), int(row['FaceRectY']),
                                int(row['FaceRectX'] + row['FaceRectWidth']), int(row['FaceRectY'] + row['FaceRectHeight'])) for _, row in frame_rows.iterrows()]
            locationsPyFeatList = []
            for feats in locationsPyFeat:
                temp = [xyz for xyz in feats]
                temp.append(1)
                locationsPyFeatList.append(temp)

            frames.append(image)
            locations_batches.append(locationsPyFeatList)

            if len(frames) == batch_size:
                if torch.cuda.is_available():
                    frames_tensor = [frame.cuda() for frame in frames]
                landmarks_batch = self.myDetector.detect_landmarks_batch(frames_tensor, locations_batches)

                for i, (frame, locations, landmarks) in enumerate(zip(frames, locations_batches, landmarks_batch)):
                    for face_location, face_landmark in zip(locations, landmarks):
                        #print(f'face_location: {face_location}\n face_landmark: {face_landmark.size}')
                        left, top, right, bottom, _ = face_location

                        loc = someDF.loc[(someDF['Frame'] == (self.frame_i - batch_size + 1 + i)) & (someDF['FaceRectY'] == top) & (someDF['FaceRectX'] == left)].index[0]
                        # Update landmark coordinates
                        for j, coords in enumerate(face_landmark):
                            cordX, cordY = coords
                            someDF.at[loc, f'x{j}'] = int(cordX)
                            someDF.at[loc, f'y{j}'] = int(cordY)
                            # someDF.at[loc, f'x{j}'] = cordX
                            # someDF.at[loc, f'y{j}'] = cordY

                frames = []
                locations_batches = []

            self.frame_i += 1

        if len(frames) > 0:
            if torch.cuda.is_available():
                frames_tensor = [frame.cuda() for frame in frames]
            landmarks_batch = self.myDetector.detect_landmarks_batch(frames_tensor, locations_batches)

            for i, (frame, locations, landmarks) in enumerate(zip(frames, locations_batches, landmarks_batch)):
                for face_location, face_landmark in zip(locations, landmarks):
                    left, top, right, bottom, _ = face_location
                    loc = someDF.loc[(someDF['Frame'] == (self.frame_i - len(frames) + i)) & (someDF['FaceRectY'] == top) & (someDF['FaceRectX'] == left)].index[0]
                    # Update landmark coordinates
                    for j, coords in enumerate(face_landmark):
                        cordX, cordY = coords
                        someDF.at[loc, f'x{j}'] = cordX
                        someDF.at[loc, f'y{j}'] = cordY
        self.video.release()
        return someDF
    
    def testFunction(self, someDF, display=False, batch_size=8):
        someDF['Person'] = pd.Series([None] * len(someDF.index))
        empty_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
        someDF = pd.concat([someDF, empty_df], axis=1)
        frame_i = 0
        total_iterator = 0
        while True:
            ret, image = self.video.read()
            if ret == False:
                break
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame_rows = someDF[someDF['Frame'] == self.frame_i]
            locationsPyFeat = [(int(row['FaceRectX']), int(row['FaceRectY']), 
                          int(row['FaceRectX'] + row['FaceRectWidth']), int(row['FaceRectY'] + row['FaceRectHeight'])) for _, row in frame_rows.iterrows()]
            locationsPyFeatList = []
            for feats in locationsPyFeat:
                temp = [xyz for xyz in feats]
                temp.append(1)
                locationsPyFeatList.append(temp)
            locationsPyFeatList = [locationsPyFeatList]
            landmarks = self.myDetector.detect_landmarks(image, locationsPyFeatList)
            
            ## top, right, bottom, left
            #landmarks = face_recognition.face_landmarks(image, locations, model=self.modelSize)
            
            for face_location, face_landmark in zip(locationsPyFeat, landmarks[0]):
                #top, right, bottom, left = face_location
                left, top, right, bottom = face_location 
                match = 1

                ## This is for visualizing the bounding boxes in real-time, with annotations
                
                loc = someDF.loc[(someDF['Frame'] == self.frame_i) & (someDF['FaceRectY'] == top) & (someDF['FaceRectX'] == left)].index[0]
                someDF.loc[loc, 'Person'] = str(match)
                
                temp_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
                
                for j, coords in enumerate(face_landmark):
                    cordX, cordY = coords
                    temp_df.at[0, f'x{j}'] = cordX
                    temp_df.at[0, f'y{j}'] = cordY
                        
                cols_to_update = temp_df.columns.intersection(someDF.columns)
                # Update only the required columns
                someDF.loc[loc, cols_to_update] = temp_df.loc[0, cols_to_update]
                
            self.frame_i += 1
            
        self.video.release()
        return someDF