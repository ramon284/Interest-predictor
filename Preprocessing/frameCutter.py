import cv2
class frameCutter:
    def __init__(self, fileName, framesToSkip=5):
        self.fileName = fileName
        self.inputPath = './Data/Video/'+fileName
        self.outputPath = './Data/Video/cut_'+fileName
        self.framesToSkip = framesToSkip
        
    def openVideo(self):    
        # Open the input video file
        input_file = self.inputPath
        self.cap = cv2.VideoCapture(input_file)

        # Get the video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def initOutput(self):
        # Create an output video file
        self.output_file = self.outputPath
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_file, self.fourcc, self.fps, (self.width, self.height), isColor=True)

    def createVideo(self):
        # Iterate through the frames
        for i in range(self.frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break
            # Only write every n-th frame to the video
            if i % self.framesToSkip == 0:
                self.out.write(frame)
        self.cap.release()
        self.out.release()
        
    def run(self):
        self.openVideo()
        self.initOutput()
        self.createVideo()
