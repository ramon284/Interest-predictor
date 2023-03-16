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
        self.initDirectory()
        
    def initDirectory(self):    
        if not os.path.exists(self.fileDir):
            os.makedirs(self.fileDir)
        for name in os.listdir(self.fileDir):
            if name == '.gitignore':
                continue
            for filename in os.listdir(f"{self.fileDir}/{name}"):
                #image = face_recognition.load_image_file(f"{KNOWN_FACES}/{filename}")
                #encoding = face_recognition.face_encodings(image)[0]
                encoding = pickle.load(open(f"{self.fileDir}/{name}/{filename}", "rb"))
                self.known_faces.append(encoding)
                self.known_names.append(int(name))

        if len(self.known_names) > 0:
            self.next_id = max(self.known_names) + 1
        else:
            self.next_id = 0

    def runDetection(self):
        while True:
            ret, image = self.video.read()
            if ret == False:
                break
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            locations = face_recognition.face_locations(image, number_of_times_to_upsample=0,  model=self.MODEL)
            encodings = face_recognition.face_encodings(image, locations, model=self.modelSize)

            for face_encoding, face_location in zip(encodings, locations):
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
                
                # top_left = (face_location[3], face_location[0])
                # bottom_right = (face_location[1], face_location[2])
                # color = [0, 255, 0]
                # cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICCNESS)
                # top_left = (face_location[3], face_location[2])
                # bottom_right = (face_location[1], face_location[2]+22)
                # cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                # cv2.putText(image, str(match), (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,200,200))
                
                top, right, bottom, left = face_location
                self.faceData = pd.concat([self.faceData, 
                                    pd.DataFrame({'Frame': self.frame_i, 'Person': str(match), 'FaceRectX': left, 'FaceRectY': top, 
                                                'FaceRectWidth': right - left, 'FaceRectHeight': bottom - top}, 
                                                index=[0])], 
                                    ignore_index=True)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('x', image)
            # if cv2.waitKey(1) & 0xFF ==ord('e'):
            #     break
            self.frame_i += 1
            
        self.video.release()
        cv2.destroyAllWindows()
        return self.faceData
    
    def saveCSV(self):
        self.faceData.to_csv('facial_persistence_test'+self.filename+'.csv', index=False)