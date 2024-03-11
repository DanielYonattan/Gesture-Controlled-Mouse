import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy
import cv2 as cv

def main(): 
    model_path = './../models/hand_landmarker.task'

    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

    # Create a hand landmarker instance with the live stream mode:
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        gesture = result.gestures[0][0].category_name if len(result.gestures) > 0 else result.gestures
        x = result.hand_landmarks[0][0].x if len(result.hand_landmarks) > 0 else result.hand_landmarks
        y = result.hand_landmarks[0][0].y if len(result.hand_landmarks) > 0 else result.hand_landmarks
        #print('hand landmarker result: {}'.format(result.gestures[0][0].category_name if len(result.gestures) > 0 else result.gestures))

   
        if gesture == "Pointing_Up":
            print("up")
        elif gesture == "Victory":
            print("not up")

        # index finger up represents mouse movement -- need to determine if index finger is up. 
        # index finger up is 'Pointing_Up' gesture
        
        # index and middle finger represents click
        # index and middle finger up is 'Victory' gesture
    
    options = GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='../model/gesture_recognizer.task'),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=print_result)

    with GestureRecognizer.create_from_options(options) as landmarker:
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

                landmarker.recognize_async(mp_image, mp.Timestamp.from_seconds(time.time()).value)
            # the 'q' button is set as the quitting button
            if cv.waitKey(1) & 0xFF == ord('q'): 
                break

        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv.destroyAllWindows() 


if __name__ == "__main__":
   main()