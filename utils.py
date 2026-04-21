import pyautogui
import Quartz.CoreGraphics as CG


screen_size = pyautogui.size()

# calculate x and y coordinates on screen given normalized coordintes
def get_xy(norm_x: int, norm_y: int):
    screen_size_x, screen_size_y = screen_size
    x = float(screen_size_x * norm_x)
    y = float(screen_size_y * norm_y)
    return x, y

def move_mouse_native(x, y):
    # Create a move event
    move_event = CG.CGEventCreateMouseEvent(
        None, 
        CG.kCGEventMouseMoved, 
        (x, y), 
        CG.kCGMouseButtonLeft
    )
    # Post the event to the system
    CG.CGEventPost(CG.kCGHIDEventTap, move_event)

def click_mouse_native(x, y):
    # Create the Mouse Down event
    down_event = CG.CGEventCreateMouseEvent(
        None, 
        CG.kCGEventLeftMouseDown, 
        (x, y), 
        CG.kCGMouseButtonLeft
    )
    
    # Create the Mouse Up event
    up_event = CG.CGEventCreateMouseEvent(
        None, 
        CG.kCGEventLeftMouseUp, 
        (x, y), 
        CG.kCGMouseButtonLeft
    )
    
    # Post them to the system
    CG.CGEventPost(CG.kCGHIDEventTap, down_event)
    CG.CGEventPost(CG.kCGHIDEventTap, up_event)