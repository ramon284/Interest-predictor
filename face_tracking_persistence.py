## Experimenting with saving face_recognition face data with pickle
## Based on some experimenting with mtcnn and retinaface models, these methods may be preferred for general face detection

import face_recognition
import os
import cv2
import pickle
import time
import numpy as np
import pandas as pd

KNOWN_FACES = "known_faces"
UNKNOWN_FACES = "unknown_faces"
TOLERANCE = 0.54 ## <- most important hyperparameter. Lower means it more quickly recognizes a "new" face
FRAME_THICCNESS = 3
FONT_THICCNESS = 2
MODEL = 'cnn' 

video = cv2.VideoCapture('Data/Video/Video0_Trim.mp4')
print(video.isOpened())

known_faces = []
known_names = []
for name in os.listdir(KNOWN_FACES):
    if name == '.gitignore':
        continue
    for filename in os.listdir(f"{KNOWN_FACES}/{name}"):
        #image = face_recognition.load_image_file(f"{KNOWN_FACES}/{filename}")
        #encoding = face_recognition.face_encodings(image)[0]
        encoding = pickle.load(open(f"{KNOWN_FACES}/{name}/{filename}", "rb"))
        known_faces.append(encoding)
        known_names.append(int(name))

if len(known_names) > 0:
    next_id = max(known_names) + 1
else:
    next_id = 0

faceData = pd.DataFrame(columns=['Frame','Person', 'FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight'])
frame_i = 0
while True:
    ret, image = video.read()
    if frame_i % 3 == 0:
        frame_i += 1
        continue
    if ret == False:
        break
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    locations = face_recognition.face_locations(image, number_of_times_to_upsample=0,  model='cnn')
    encodings = face_recognition.face_encodings(image, locations, model='large')

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f"{KNOWN_FACES}\{match}")
            pickle.dump(face_encoding, open(f"{KNOWN_FACES}\{match}\{match}-{int(time.time())}.pkl", 'wb'))

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
        faceData = pd.concat([faceData, 
                            pd.DataFrame({'Frame': frame_i, 'Person': str(match), 'FaceRectX': left, 'FaceRectY': top, 
                                          'FaceRectWidth': right - left, 'FaceRectHeight': bottom - top}, 
                                         index=[0])], 
                            ignore_index=True)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('x', image)
    # if cv2.waitKey(1) & 0xFF ==ord('e'):
    #     break
    frame_i += 1
    
video.release()
cv2.destroyAllWindows()
faceData.to_csv('facial_persistence_test.csv', index=False)