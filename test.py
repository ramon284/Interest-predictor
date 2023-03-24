import numpy as np
import pandas as pd
from Models.tracking_persistence.face_tracking_persistence import face_persistence_model
from Models.tracking_persistence.face_tracking_persistence import face_persistence_model
from Models.face_detection.face_detection import face_detector
from feat.au_detectors.StatLearning.SL_test import XGBClassifier
import os
import cv2
import pickle
import time
import numpy as np
import pandas as pd

filename = 'Video2_Trim_Trim.mp4'

face_detector_model = face_detector(filename=filename, detection_threshold=0.25,)
output = face_detector_model.runModel()
face_tracker = face_persistence_model(TOLERANCE=0.6, filename=filename, model='cnn', modelSize='large')
output = face_tracker.runDetectionTrackOnly(output, True)
face_tracker.saveCSV(output)


