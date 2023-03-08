## Experimenting with saving face_recognition face data with pickle
## Based on some experimenting with mtcnn and retinaface models, these methods may be preferred for general face detection

import face_recognition
import os
import cv2
import pickle
import time

KNOWN_FACES = "known_faces"
UNKNOWN_FACES = "unknown_faces"
TOLERANCE = 0.5
FRAME_THICCNESS = 3
FONT_THICCNESS = 2
MODEL = 'cnn' 

video = cv2.VideoCapture('Data/Video/Video0.mp4')
print(video.isOpened())

known_faces = []
known_names = []
for name in os.listdir(KNOWN_FACES):
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

while True:
    ret, image = video.read()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    locations = face_recognition.face_locations(image, number_of_times_to_upsample=0,  model='hog')
    encodings = face_recognition.face_encodings(image, locations)

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

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = [0, 255, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICCNESS)
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, str(match), (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,200,200))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('x', image)
    if cv2.waitKey(1) & 0xFF ==ord('e'):
        break
video.release()
cv2.destroyAllWindows()
