import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy
import cv2 as cv
import pyautogui
import threading
import queue
from collections import deque
from utils import get_xy, move_mouse_native, click_mouse_native
import faulthandler
faulthandler.enable()

last_click_timestamp = 0
COOLDOWN = 500

frame_queue = queue.Queue(maxsize=1)

mouse_lock = threading.Lock()
latest_gesture = {"gesture": None, "x": None, "y": None, "timestamp_ms": 0}

GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

def move_mouse(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        global last_click_timestamp 
    
        if not result.hand_landmarks or not result.gestures:
            return

        gesture = result.gestures[0][0].category_name if len(result.gestures) > 0 else result.gestures
        x = result.hand_landmarks[0][0].x if len(result.hand_landmarks) > 0 else result.hand_landmarks
        y = result.hand_landmarks[0][0].y if len(result.hand_landmarks) > 0 else result.hand_landmarks

        x, y = get_xy(x, y)

        with mouse_lock:
            latest_gesture["gesture"] = gesture
            latest_gesture["x"] = x
            latest_gesture["y"] = y
            latest_gesture["timestamp_ms"] = timestamp_ms

def capture_frames(cap: cv.VideoCapture):
    while True:
        frame_exists, frame = cap.read()

        if not frame_exists:
            break
        
        frame = cv.flip(frame, 1) # mirror image

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

def dispatch_mouse_events():
    global last_click_timestamp

    while True:
        with mouse_lock:
            gesture = latest_gesture["gesture"]
            x = latest_gesture["x"]
            y = latest_gesture["y"]
            timestamp_ms = latest_gesture["timestamp_ms"]
        
        if gesture and x is not None and y is not None:
            if gesture == "Pointing_Up":
                move_mouse_native(x, y)
            elif gesture == "Victory":
                if (timestamp_ms - last_click_timestamp) >= COOLDOWN:
                    move_mouse_native(x, y)
                    click_mouse_native(x, y)
                    last_click_timestamp = timestamp_ms
        
        time.sleep(0.005)


def main(): 
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    
    options = GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='./model/gesture_recognizer.task'),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=move_mouse)

    cap = cv.VideoCapture(0)
    
    capture_thread = threading.Thread(target=capture_frames, args=(cap,), daemon=True)
    mouse_thread = threading.Thread(target=dispatch_mouse_events, daemon=True)

    capture_thread.start()
    mouse_thread.start()

    with GestureRecognizer.create_from_options(options) as recognizer:
        while(True): 
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)

            cv.imshow('frame', frame) 

            # the 'q' button is set as the quitting button
            if cv.waitKey(1) & 0xFF == ord('q'): 
                break

        cap.release() 
        cv.destroyAllWindows() 


if __name__ == "__main__":
   main()