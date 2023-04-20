import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
import math

class mediaPipeModel:
    def __init__(self, filename, df, display=False):
        self.filename = filename
        self.df = df
        self.df['pupil_ratio'] = pd.Series(dtype='float')
        self.df['pupil_direction'] = pd.Series(dtype='object')
        self.bbox_columns = ['FaceRectX', 'FaceRectWidth', 'FaceRectY', 'FaceRectHeight']
        self.blue = [255, 255, 0]
        self.cyan =  [255, 0, 0]
        self.red = [0,0,255]
        self.display = display
        self.RIGHT_IRIS = [474, 475, 476, 477]
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.L_H_LEFT = [33]  # right eye right most landmark
        self.L_H_RIGHT = [133]  # right eye left most landmark
        self.R_H_LEFT = [362]  # left eye right most landmark
        self.R_H_RIGHT = [263]  # left eye left most landmark
        self.max_num_faces = 1
        self.refine_landmarks=True,
        self.min_detection_confidence=0.5,
        self.min_tracking_confidence=0.5
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        

        
        ## self.mp_face_mesh = mp.solutions.face_mesh
        ## self.mp_drawing = mp.solutions.drawing_utils
        
        
    def euclidian_distance(self, p1, p2):
        x1, y1 = p1.ravel()
        x2, y2 = p2.ravel()
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance

    def irisPosition(self, iris_center, right_point, left_point):
        center_to_right_distance = self.euclidian_distance(iris_center, right_point)
        #center_to_left_distance = euclidian_distance(iris_center, left_point)
        total_distance = self.euclidian_distance(right_point, left_point)
        ratio = center_to_right_distance/total_distance
        iris_position = ''
        if ratio <= 0.45:
            iris_position = 'right'
        elif ratio >= 0.45 and ratio <= 0.54:
            iris_position = 'center'
        else:
            iris_position = 'left'
        return iris_position, ratio
    
    def loadVideo(self):
        self.video = cv2.VideoCapture('./Data/Video/'+self.filename)
        
    def saveCSV(self):
        self.df.to_csv(f'mediapipe_{self.filename}.csv', index=False)
        
    def runModel(self):
        for person in self.df.Person.unique():
            self.loadVideo()
            frame_i = 0
            total_iterator = 0
            temp_df = self.df.loc[self.df['Person'] == person]
            # with self.mp_face_mesh.FaceMesh(
            #     max_num_faces=1,
            #     refine_landmarks=True,
            #     min_detection_confidence=0.5,
            #     min_tracking_confidence=0.5
            # ) as face_mesh:
            while True:
                ret, image = self.video.read()
                if ret == False:
                    break
                
                if frame_i not in temp_df['Frame'].values:
                    frame_i += 1
                    continue    
                else: ## get bounding box first
                    frame_row = temp_df[temp_df['Frame'] == frame_i]
                    left, width, top, height = frame_row.iloc[0][self.bbox_columns]  
                    # print(left, width, top, height)
                    left = int(left)
                    right = int(left + width)
                    top = int(top)
                    bottom = int(top + height)                    
                    results = self.face_mesh
                    face_frame = image[top:bottom, left:right] ## this is the bounding box of a face
                    img_h, img_w = face_frame.shape[:2]
                    results = self.face_mesh.process(face_frame)
                    
                    if results.multi_face_landmarks:
                        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
                        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])
                        center_left = np.array([l_cx, l_cy], dtype=np.int32)
                        center_right = np.array([r_cx, r_cy], dtype=np.int32)
                        top_left = (top, left)
                        bottom_right = (bottom, right)
                        center_left_full = tuple(a + b for a, b in zip(center_left, top_left))
                        center_right_full = tuple(a + b for a, b in zip(center_right, top_left))

                        if self.display:
                            cv2.circle(face_frame, center_left, int(l_radius), (255,0,255), 1, cv2.LINE_AA)
                            cv2.circle(face_frame, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)
                            cv2.circle(face_frame, mesh_points[self.R_H_RIGHT][0], 2, (255,255,255), 2, cv2.LINE_AA)
                            cv2.circle(face_frame, mesh_points[self.R_H_LEFT][0], 2, (0,255,255), 2, cv2.LINE_AA)
                        iris_pos, ratio = self.irisPosition(center_right, mesh_points[self.R_H_RIGHT][0], mesh_points[self.R_H_LEFT][0])
                        # print(f'right eye left side = {left + mesh_points[self.R_H_LEFT][0][0]}, right eye right side = {left + mesh_points[self.R_H_RIGHT][0][0]}\n' +
                        #     f'left eye left side = {left + mesh_points[self.L_H_LEFT][0][0]}, left eye right side = {left + mesh_points[self.L_H_RIGHT][0][0]}\n', 
                        #     f'iris and ratio = {iris_pos, ratio}')
                        
                        row_index = (self.df['Frame'] == frame_i) & (self.df['Person'] == person)
                        # Update the 'pupil_ratio' and 'pupil_direction' values in self.df
                        self.df.loc[row_index, 'pupil_ratio'] = ratio
                        self.df.loc[row_index, 'pupil_direction'] = iris_pos
                    else:
                        pass
                        
                    if self.display:
                        frame_i += 1  
                        cv2.circle(face_frame, center_left + (left, top), int(l_radius), (255,0,255), 1, cv2.LINE_AA)
                        cv2.imshow('img', image)
                        if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                            self.video.release()
                            cv2.destroyAllWindows()
                            break
                    # Wait for 1/fps seconds before displaying the next frame
                    else:
                        frame_i += 1   
        return self.df
        #self.saveCSV()
                    
                                  