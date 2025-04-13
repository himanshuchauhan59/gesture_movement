import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

# Initialize
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize previous position
prev_x, prev_y = 0, 0
smoothening = 5  # Higher value = more smoothness but more delay
is_dragging = False
drag_start_time = 0
click_threshold_z = 0.03  # Threshold for click detection
pinch_threshold = 0.05  # Threshold for pinch detection
drag_delay = 0.3  # Seconds of pinch needed to start drag

def move_mouse_smooth(x, y):
    global prev_x, prev_y
    # Smooth the movement
    curr_x = prev_x + (x - prev_x) / smoothening
    curr_y = prev_y + (y - prev_y) / smoothening
    
    pyautogui.moveTo(curr_x, curr_y)
    
    prev_x, prev_y = curr_x, curr_y

while True:
    success, img = cap.read()
    if not success:
        continue
        
    img = cv2.flip(img, 1)  # Mirror the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get finger tips
            tip_ids = [4, 8, 12, 16, 20]
            
            first_finger = hand_landmarks.landmark[tip_ids[1]]
            thumb = hand_landmarks.landmark[tip_ids[0]]
            second_finger = hand_landmarks.landmark[tip_ids[2]]
            
            
            # Convert to screen coordinates
            screen_width, screen_height = pyautogui.size()
            first_finger_x = int(first_finger.x * screen_width)
            first_finger_y = int(first_finger.y * screen_height)
            second_finger_x = int(second_finger.x * screen_width)
            second_finger_y = int(second_finger.y * screen_height)
            
            # Calculate pinch distance (normalized)
            pinch_distance = abs(first_finger.y - thumb.y)
            
            # Detect pinch gesture
            pinch_detected = pinch_distance <= pinch_threshold
            
            # Drag logic
            if pinch_detected:
                if not is_dragging:
                    # Start timing the pinch
                    if drag_start_time == 0:
                        drag_start_time = time.time()
                    # If pinch held long enough, start dragging
                    elif time.time() - drag_start_time > drag_delay:
                        is_dragging = True
                        pyautogui.mouseDown(button='left')
                        print("Drag started")
            else:
                # Reset drag timer if not pinching
                drag_start_time = 0
                if is_dragging:
                    is_dragging = False
                    pyautogui.mouseUp(button='left')
                    print("Drag ended")

            # Move the mouse (whether dragging or not)
            move_mouse_smooth(first_finger_x, first_finger_y)
            
            print(f"Finger position: {abs(abs(first_finger.x)- abs(second_finger.x))}, {abs(abs(first_finger.x)- abs(second_finger.x)) <= click_threshold_z}")
            
            
            # # Click detection (based on z-distance)
            if abs(abs(first_finger.x)- abs(second_finger.x)) <= click_threshold_z and not is_dragging:
                pyautogui.click(first_finger_x, first_finger_y)
                print(f"Click at {first_finger_x}, {first_finger_y}")
                # Small delay to prevent multiple clicks
                time.sleep(0.2)

            # Display debug info
            cv2.putText(
                img,                              
                f"Z: {abs(first_finger.z):.2f}",     
                (50, 50),                         
                cv2.FONT_HERSHEY_SIMPLEX,         
                1,                                
                (0, 255, 0),                      
                2                                 
            )
            cv2.putText(
                img,                              
                f"Pinch: {pinch_distance:.2f}",     
                (50, 100),                         
                cv2.FONT_HERSHEY_SIMPLEX,         
                1,                                
                (0, 255, 0),                      
                2                                 
            )
            cv2.putText(
                img,                              
                f"State: {'DRAG' if is_dragging else 'MOVE'}",     
                (50, 150),                         
                cv2.FONT_HERSHEY_SIMPLEX,         
                1,                                
                (0, 255, 0),                      
                2                                 
            )

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
if is_dragging:
    pyautogui.mouseUp(button='left')
cap.release()
cv2.destroyAllWindows()