import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Box position and size
box_size = 100
box_color = (0, 255, 0)  # Green box color
box_thickness = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates (Landmark 8)
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert to pixel coordinates
            h, w, _ = frame.shape
            index_finger_x = int(index_finger_tip.x * w)
            index_finger_y = int(index_finger_tip.y * h)

            # Check if the index finger is extended (simple threshold based on y-axis)
            # Assuming the thumb is down and index is up, check if y-value of tip is higher than others
            if index_finger_tip.y < landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y:
                # Draw a box where the finger tip is
                top_left = (index_finger_x - box_size // 2, index_finger_y - box_size // 2)
                bottom_right = (index_finger_x + box_size // 2, index_finger_y + box_size // 2)
                cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)

    # Display the frame
    cv2.imshow('Hand Tracking with Box', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
