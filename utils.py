import Quartz.CoreGraphics as CG

def get_screen_size():
    bounds = CG.CGDisplayBounds(CG.CGMainDisplayID())
    return bounds.size.width, bounds.size.height

screen_size = get_screen_size()
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

    if move_event is None:
        return
    # Post the event to the system
    CG.CGEventPost(CG.kCGSessionEventTap, move_event) 


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
    
    
    if down_event:
        CG.CGEventPost(CG.kCGSessionEventTap, down_event)
    
    if up_event:
        CG.CGEventPost(CG.kCGSessionEventTap, up_event)
