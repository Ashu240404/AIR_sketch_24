import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Drawing canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
positions = deque(maxlen=5)

mode = "idle"
menu_open = False
menu_triggered_once = False
last_x, last_y = None, None
shape_offset = [0, 0]
draw_color = (0, 0, 255)  # Default red
selected_message = ""
message_time = 0

color_blocks = {
    'red': ((580, 50), (630, 100), (0, 0, 255)),
    'green': ((580, 110), (630, 160), (0, 255, 0)),
    'blue': ((580, 170), (630, 220), (255, 0, 0)),
    'yellow': ((580, 230), (630, 280), (0, 255, 255))
}

# Helper: Count fingers
def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

prev_x, prev_y = None, None  # For grab
grab_offset = [0, 0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger = hand_landmarks.landmark[8]
            cx, cy = int(index_finger.x * w), int(index_finger.y * h)

            positions.append((cx, cy))
            avg_x = int(np.mean([p[0] for p in positions]))
            avg_y = int(np.mean([p[1] for p in positions]))

            fingers_up = count_fingers(hand_landmarks)

            # Trigger menu only once
            if fingers_up == 3 and not menu_triggered_once:
                menu_open = True
                menu_triggered_once = True
                mode = "menu"
                last_x, last_y = None, None

            if menu_open and fingers_up == 1:
                for name, ((x1, y1), (x2, y2), color) in color_blocks.items():
                    if x1 < avg_x < x2 and y1 < avg_y < y2:
                        draw_color = color
                        selected_message = f"{name.capitalize()} Color Selected"
                        message_time = current_time
                        menu_open = False  # Close menu after selection
                        mode = "idle"

            if not menu_open:
                if fingers_up == 1:  # Draw mode
                    mode = "draw"
                    if last_x is None or last_y is None:
                        last_x, last_y = avg_x, avg_y
                    cv2.line(canvas, (last_x, last_y), (avg_x, avg_y), draw_color, 5)
                    last_x, last_y = avg_x, avg_y

                elif fingers_up == 0:  # Fist → Grab mode
                    mode = "grab"
                    if prev_x is None or prev_y is None:
                        prev_x, prev_y = avg_x, avg_y

                    dx = avg_x - prev_x
                    dy = avg_y - prev_y

                    grab_offset[0] += dx
                    grab_offset[1] += dy

                    prev_x, prev_y = avg_x, avg_y

                elif fingers_up == 5:  # Open palm → Erase
                    mode = "erase"
                    canvas = np.zeros_like(canvas)
                    grab_offset = [0, 0]
                    last_x, last_y = None, None
                    prev_x, prev_y = None, None

                else:
                    mode = "idle"
                    last_x, last_y = None, None
                    prev_x, prev_y = None, None

    # Apply grab offset
    moved_canvas = np.zeros_like(canvas)
    M = np.float32([[1, 0, grab_offset[0]], [0, 1, grab_offset[1]]])
    moved_canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))

    combined = cv2.addWeighted(frame, 0.5, moved_canvas, 0.5, 0)

    # Draw color menu if open
    if menu_open:
        for name, ((x1, y1), (x2, y2), color) in color_blocks.items():
            cv2.rectangle(combined, (x1, y1), (x2, y2), color, -1)
            cv2.putText(combined, name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Display selected message
    if selected_message and (current_time - message_time < 2):
        cv2.putText(combined, selected_message, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    elif current_time - message_time >= 2:
        selected_message = ""

    cv2.putText(combined, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Hand Draw & Grab & Erase', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
