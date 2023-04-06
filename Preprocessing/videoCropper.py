import cv2

class videoCropper:
    def __init__(self, df, videoFileName, displayVideo, saveCSV=True):
        self.df = df.copy(deep=True)
        self.path = './Data/Video/'
        self.videoFileName = videoFileName
        self.displayVideo = displayVideo
        self.saveCSV = saveCSV
        
    def openVideo(self):
        self.cap = cv2.VideoCapture(self.path + self.videoFileName)
        
    def readDataFrame(self):
        ## Find the extreme locations of the bounding boxes in the entire video
        ## By looking at every bounding box in every frame
        sum_cols_low = ['FaceRectY', 'FaceRectHeight']
        sum_cols_right = ['FaceRectX', 'FaceRectWidth']
        self.left = int(self.df['FaceRectX'].min())
        self.right = int(self.df[sum_cols_right].sum(axis=1).max())
        self.top = int(self.df['FaceRectY'].min())
        self.bottom = int(self.df[sum_cols_low].sum(axis=1).max())
    
    def getVideoProperties(self):
    # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.top = max(0, self.top)
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.bottom = min(frame_height, self.bottom)
        self.width = int(self.right - self.left)
        self.height = int(self.bottom - self.top)
        print(f'right:{self.right} and left:{self.left}')

    def createCroppedVideo(self):
    # Create a VideoWriter to save the cropped video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs, like 'XVID' for .avi files
        output_video = './Data/Video/cropped_'+self.videoFileName
        out = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))
        print(self.top, self.bottom, self.height, self.width)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            cropped_frame = frame[self.top:self.bottom, self.left:self.right]
            # Save the cropped frame to the new video file
            out.write(cropped_frame)

            # Display the cropped frame (optional)
            if(self.displayVideo == True):
                cv2.imshow('Cropped Frame', cropped_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

        self.df['FaceRectX'] = self.df['FaceRectX'] - self.left
        self.df['FaceRectY'] = self.df['FaceRectY'] - self.top

    def saveNewCsv(self, csvName='updated_dataframe'):
        self.df.to_csv(csvName+'.csv', index=False)
    
    def runModel(self, csvName='updated_dataframe'):
        self.openVideo()
        self.readDataFrame()
        self.getVideoProperties()
        self.createCroppedVideo()
        if self.saveCSV:
            self.saveNewCsv()
        return self.df
        
        
