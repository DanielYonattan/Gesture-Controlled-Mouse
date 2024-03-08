import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy
import cv2 as cv

def main(): 
    model_path = './../models/hand_landmarker.task'

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

    # Create a hand landmarker instance with the live stream mode:
    def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        print('hand landmarker result: {}'.format(result))

    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='../hand-tracker/model/hand_landmarker.task'),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=print_result)

    with HandLandmarker.create_from_options(options) as landmarker:
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
        vid = cv.VideoCapture(0)

        while(True): 
            # Capture the video frame by frame 
            frame_exists, frame = vid.read() 

            if frame_exists:            
                # Display the resulting frame 
                cv.imshow('frame', frame) 

                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                detection_result = landmarker.detect_async(mp_image, mp.Timestamp.from_seconds(time.time()).value)
            # the 'q' button is set as the quitting button
            if cv.waitKey(1) & 0xFF == ord('q'): 
                break

        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv.destroyAllWindows() 


if __name__ == "__main__":
   main()