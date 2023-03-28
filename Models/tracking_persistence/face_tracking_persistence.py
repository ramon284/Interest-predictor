## Experimenting with saving face_recognition face data with pickle
## Based on some experimenting with mtcnn and retinaface models, these methods may be preferred for general face detection

import face_recognition
import os
import cv2
import pickle
import time
import numpy as np
import pandas as pd

class face_persistence_model:
    def __init__(self, TOLERANCE, filename, model='cnn', modelSize='large'):
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
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_rows = someDF[someDF['Frame'] == self.frame_i]
            locations = [(int(row['FaceRectY']), int(row['FaceRectX'] + row['FaceRectWidth']), 
                          int(row['FaceRectHeight'] + row['FaceRectY']), int(row['FaceRectX'])) for _, row in frame_rows.iterrows()]
            ## top, right, bottom, left
            encodings = face_recognition.face_encodings(image, locations, model=self.modelSize)
            landmarks = face_recognition.face_landmarks(image, locations, model=self.modelSize)
            for face_encoding, face_location, face_landmark in zip(encodings, locations, landmarks):
                top, right, bottom, left = face_location
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
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    color = [255, 255, 0]
                    cv2.rectangle(image, top_left, bottom_right, color, self.FRAME_THICCNESS)
                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2]+22)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, str(match), (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0))
                
                loc = someDF.loc[(someDF['Frame'] == self.frame_i) & (someDF['FaceRectY'] == top) & (someDF['FaceRectX'] == left)].index[0]
                someDF.loc[loc, 'Person'] = str(match)
                temp_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
                for landmark, indices in self.landmark_dict.items():
                    for index, (x,y) in zip(indices, face_landmark[landmark]):
                        temp_df.at[0, f'x{index}'] = x
                        temp_df.at[0, f'y{index}'] = y
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
            
            
    def WORKINPROGRESS(self, someDF, display=False):
        someDF['Person'] = pd.Series([None] * len(someDF.index))
        empty_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
        someDF = pd.concat([someDF, empty_df], axis=1)
        frame_i = 0
        total_iterator = 0
        self.auDF = pd.DataFrame(index=range(len(someDF)), columns=self.au_keys)
        self.emoDF = pd.DataFrame(index=range(len(someDF)), columns=self.emotion_keys)
        while True:
            ret, image = self.video.read()
            if ret == False:
                break
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_rows = someDF[someDF['Frame'] == self.frame_i]
            locations = [(int(row['FaceRectY']), int(row['FaceRectX'] + row['FaceRectWidth']), 
                          int(row['FaceRectHeight'] + row['FaceRectY']), int(row['FaceRectX'])) for _, row in frame_rows.iterrows()]
            ## top, right, bottom, left
            encodings = face_recognition.face_encodings(image, locations, model=self.modelSize)
            landmarks = face_recognition.face_landmarks(image, locations, model=self.modelSize)
            for face_encoding, face_location, face_landmark in zip(encodings, locations, landmarks):
                top, right, bottom, left = face_location
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
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    color = [255, 255, 0]
                    cv2.rectangle(image, top_left, bottom_right, color, self.FRAME_THICCNESS)
                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2]+22)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, str(match), (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0))
                
                loc = someDF.loc[(someDF['Frame'] == self.frame_i) & (someDF['FaceRectY'] == top) & (someDF['FaceRectX'] == left)].index[0]
                someDF.loc[loc, 'Person'] = str(match)
                temp_df = pd.DataFrame(columns=[f'x{i}' for i in range(68)] + [f'y{i}' for i in range(68)])
                for landmark, indices in self.landmark_dict.items():
                    for index, (x,y) in zip(indices, face_landmark[landmark]):
                        temp_df.at[0, f'x{index}'] = x
                        temp_df.at[0, f'y{index}'] = y
                x_array = temp_df.iloc[0, self.x_columns].tolist()
                y_array = temp_df.iloc[0, self.y_columns].tolist()
                coordinates = list(zip(x_array, y_array))
                coordinates = np.array(coordinates, dtype=np.float32)
                nested_coordinates = [[coordinates]]
                print(nested_coordinates)
                aus = self.detectorModel.detect_aus(image, nested_coordinates)
                if self.emoModel == 'svm':
                    emotions = self.detectorModel.detect_emotions(frame=image, facebox=['a','b'], landmarks=nested_coordinates)
                # else:
                #     faceInfo = row[self.bbox_columns].tolist()
                #     faceInfo.append(1)
                #     faceInfo = [int(x) for x in faceInfo]
                #     faceInfo = [[faceInfo]]
                #     emotions = self.detectorModel.detect_emotions(frame=image, facebox=faceInfo, landmarks=nested_coordinates)
                for name, key in zip(self.au_keys,aus[0][0]):
                    self.auDF.loc[total_iterator, name] = key
                for name, key in zip(self.emotion_keys, emotions[0][0]):
                    self.emoDF.loc[total_iterator, name] = key
                total_iterator += 1
                
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
        self.someDF = pd.concat([self.someDF, self.auDF, self.emoDF], axis=1)
        self.someDF.to_csv('final_new_test_csv', index=False)
        return someDF