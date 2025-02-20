import cv2

#capture 

#select computer's camera
capture = cv2.VideoCapture(0) # 0 (zero) means => computer's default camera.

#get width of captured video as integer
capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#get height of captured video as integer
capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Width:  {capture_width}  Height: {capture_height}")

#recording video for macOS.
# 20 => fps
#fourcc is a char code that convert your video to ???.
writer = cv2.VideoWriter('trial-video-open-cv.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (capture_width, capture_height))

def initalizeKeyBinds()->bool:          # if you dont call this func you can not show your images with opencv
    keybind = cv2.waitKey(1) &0xFF      # 0xFF converts & marks the keybinds & range of 0 - 255 / q means = 113
                                        # if you set waitKey to Zero, your keyboard actions gonna listen until you press a keybind, so we set it to 1
    if keybind == ord('q'):             # returns 113 for q value
        cv2.destroyAllWindows()
        return True
    return False
   


while True:
    ret,frame = capture.read()
    cv2.imshow("Video",frame)
    cv2.rectangle(frame,(512,512),(1024,1024),(0,255,0),cv2.FILLED) #(image, (start point), (end point), rectangle_width)
    cv2.imshow("Video",frame)
    if(not ret):
        break

    #record (writer)
    writer.write(frame)

    if initalizeKeyBinds():
        break

capture.release()
writer.release()
cv2.destroyAllWindows()

  






