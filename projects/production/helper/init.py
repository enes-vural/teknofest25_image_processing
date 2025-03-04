import cv2
import constants.constants as const


class Initalize:
    video = cv2.VideoCapture(const.video_path)

    def __init__(self):
        print("Initalize class is created")

    def run(self)->None:

        frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.video.get(cv2.CAP_PROP_FPS))

        print("Video Properties")
        print("----------------------")
        print("Width: ",frame_width)
        print("Height: ",frame_height)
        print("Frame Count: ",frame_count)
        print("FPS: ",fps)
        print("----------------------")

    # check if the video is opened
    def checkVideoStartState(self)->None:
        if not self.video.isOpened():
            print("Error: Could not open video.")
            exit()

    # if 'q' is pressed, close the window
    def initalizeKeyBinds(self)->bool:  
        keybind = cv2.waitKey(1) &0xFF 
        if keybind == ord('q'):
            cv2.destroyAllWindows()
            return True
        return False
    
    def replayVideo(self)->None:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def ifVideoEnded(self,state)->None:
        if not state:
            print("Video Ended")
            self.replayVideo()
            return True
        return False
