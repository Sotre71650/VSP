import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
import logging

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(filename='hand_tracking_debug.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# -----------------------------
# Frame Capture Class
# -----------------------------
class VideoCaptureThread:
    def __init__(self, src=0, width=640, height=480):
        # Use CAP_DSHOW for Windows to improve camera access speed; remove if on Linux/macOS
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# -----------------------------
# Initialization
# -----------------------------

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize PyAutoGUI
pyautogui.FAILSAFE = False  # Disable fail-safe to prevent exceptions when moving the mouse to screen corners
pyautogui.PAUSE = 0          # Remove any pause between PyAutoGUI actions

# Get screen size
screen_width, screen_height = pyautogui.size()
logging.info(f"Screen size: {screen_width}x{screen_height}")

# Parameters for pinch detection
PINCH_THRESHOLD = 30               # Distance threshold for two-finger pinch (thumb and index) - Reduced for higher precision
THREE_FINGER_PINCH_THRESHOLD = 30  # Distance threshold for three-finger pinch - Reduced for higher precision
HOLD_THRESHOLD_FRAMES = 5          # Number of consecutive frames to hold the mouse down
CLICK_DEBOUNCE_TIME = 0.3          # Minimum time between clicks in seconds

# Smoothing and sensitivity parameters
SMOOTHING_FACTOR = 0.05             # Lowered Exponential Moving Average factor for increased responsiveness
sensitivity = 6.0                    # Slightly reduced sensitivity factor for more controlled cursor movement
MAX_MOVE_DELTA = 100                 # Maximum pixels the cursor can move per frame
NEUTRAL_ZONE_SIZE = 20              # Reduced dead zone for better responsiveness

prev_move_x, prev_move_y = 0, 0    # Previous movement deltas

# Initialize variables for pinch detection
pinch_count = 0
holding_left = False
last_left_click_time = 0

# Initialize variables for three-finger pinch
three_pinch_count = 0
holding_right = False
last_right_click_time = 0

# Performance Optimization: Resize frame for faster processing
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Initialize Frame Capture Thread
video_capture = VideoCaptureThread(src=0, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()
logging.info("Frame capture thread started.")

# Initialize Mediapipe Hands
with mp_hands.Hands(
    max_num_hands=1,               # Limiting to one hand for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    # Initialize previous finger position
    prev_index_x, prev_index_y = None, None

    # Frame rate calculation variables
    prev_time = 0
    fps = 0

    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            logging.warning("Failed to read frame.")
            continue

        # Flip the frame horizontally for a mirror view and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Get frame dimensions
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame (optional: can be commented out to save processing)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of thumb tip, index finger tip, and middle finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Convert normalized coordinates to pixel values
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

                # Draw circles on thumb tip, index finger tip, and middle finger tip (optional)
                # cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
                # cv2.circle(frame, (index_x, index_y), 8, (255, 0, 0), -1)
                # cv2.circle(frame, (middle_x, middle_y), 8, (255, 0, 0), -1)

                # Calculate Euclidean distances
                distance_thumb_index = math.hypot(index_x - thumb_x, index_y - thumb_y)
                distance_thumb_middle = math.hypot(middle_x - thumb_x, middle_y - thumb_y)

                # --- Cursor Movement ---
                if prev_index_x is not None and prev_index_y is not None:
                    # Calculate movement delta
                    delta_x = index_x - prev_index_x
                    delta_y = index_y - prev_index_y

                    # Update previous positions
                    prev_index_x, prev_index_y = index_x, index_y

                    # Apply scaling factor to movement delta
                    move_x = delta_x * sensitivity
                    move_y = delta_y * sensitivity

                    # Apply dead zone
                    if abs(move_x) < NEUTRAL_ZONE_SIZE:
                        move_x = 0
                    if abs(move_y) < NEUTRAL_ZONE_SIZE:
                        move_y = 0

                    # Cap movement delta to prevent large jumps
                    move_x = max(min(move_x, MAX_MOVE_DELTA), -MAX_MOVE_DELTA)
                    move_y = max(min(move_y, MAX_MOVE_DELTA), -MAX_MOVE_DELTA)

                    # Apply Exponential Moving Average (EMA) for smoothing
                    move_x = prev_move_x + (move_x - prev_move_x) * SMOOTHING_FACTOR
                    move_y = prev_move_y + (move_y - prev_move_y) * SMOOTHING_FACTOR

                    # Move the mouse cursor relative to its current position
                    try:
                        pyautogui.moveRel(move_x, move_y, duration=0)
                        logging.info(f"Cursor moved: ({move_x:.2f}, {move_y:.2f})")
                        print(f"Cursor moved: ({move_x:.2f}, {move_y:.2f})")
                    except Exception as e:
                        logging.error(f"Error moving cursor: {e}")
                        print(f"Error moving cursor: {e}")

                    # Update previous movement deltas
                    prev_move_x, prev_move_y = move_x, move_y

                    # Display movement deltas for debugging (optional)
                    cv2.putText(frame, f'Delta X: {int(delta_x)}', (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f'Delta Y: {int(delta_y)}', (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    # Initialize previous positions
                    prev_index_x, prev_index_y = index_x, index_y
                    logging.info("Initializing previous positions.")
                    print("Initializing previous positions.")

                # --- Three-Finger Pinch Detection for Right Click ---
                # Enhanced precision by requiring all three fingers to be very close
                if distance_thumb_index < THREE_FINGER_PINCH_THRESHOLD and distance_thumb_middle < THREE_FINGER_PINCH_THRESHOLD:
                    three_pinch_count += 1
                    logging.debug(f"Three-Finger Pinch Count: {three_pinch_count}")
                    if not holding_right and three_pinch_count >= HOLD_THRESHOLD_FRAMES:
                        try:
                            pyautogui.mouseDown(button='right')
                            holding_right = True
                            logging.info("Right mouse button held down.")
                            print("Right mouse button held down.")
                        except Exception as e:
                            logging.error(f"Error performing right mouse down: {e}")
                            print(f"Error performing right mouse down: {e}")
                        # Visual Feedback (optional)
                        cv2.putText(frame, 'Holding Right Click...', (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if holding_right:
                        try:
                            pyautogui.mouseUp(button='right')
                            holding_right = False
                            logging.info("Right mouse button released.")
                            print("Right mouse button released.")
                        except Exception as e:
                            logging.error(f"Error performing right mouse up: {e}")
                            print(f"Error performing right mouse up: {e}")
                        # Visual Feedback (optional)
                        cv2.putText(frame, 'Released Right Click', (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    elif three_pinch_count > 0:
                        # Perform a quick right click with debounce
                        current_time = time.time()
                        if current_time - last_right_click_time > CLICK_DEBOUNCE_TIME:
                            try:
                                pyautogui.click(button='right')
                                last_right_click_time = current_time
                                logging.info("Right mouse clicked.")
                                print("Right mouse clicked.")
                            except Exception as e:
                                logging.error(f"Error performing right mouse click: {e}")
                                print(f"Error performing right mouse click: {e}")
                            # Visual Feedback (optional)
                            cv2.putText(frame, 'Right Click', (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        three_pinch_count = 0

                # --- Two-Finger Pinch Detection for Left Click ---
                # Enhanced precision by ensuring two fingers are very close and avoiding accidental overlaps
                if not (distance_thumb_index < THREE_FINGER_PINCH_THRESHOLD and distance_thumb_middle < THREE_FINGER_PINCH_THRESHOLD):
                    if distance_thumb_index < PINCH_THRESHOLD:
                        pinch_count += 1
                        logging.debug(f"Two-Finger Pinch Count: {pinch_count}")
                        if not holding_left and pinch_count >= HOLD_THRESHOLD_FRAMES:
                            try:
                                pyautogui.mouseDown(button='left')
                                holding_left = True
                                logging.info("Left mouse button held down.")
                                print("Left mouse button held down.")
                            except Exception as e:
                                logging.error(f"Error performing left mouse down: {e}")
                                print(f"Error performing left mouse down: {e}")
                            # Visual Feedback (optional)
                            cv2.putText(frame, 'Holding Left Click...', (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        if holding_left:
                            try:
                                pyautogui.mouseUp(button='left')
                                holding_left = False
                                logging.info("Left mouse button released.")
                                print("Left mouse button released.")
                            except Exception as e:
                                logging.error(f"Error performing left mouse up: {e}")
                                print(f"Error performing left mouse up: {e}")
                            # Visual Feedback (optional)
                            cv2.putText(frame, 'Released Left Click', (10, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        elif pinch_count > 0:
                            # Perform a quick click with debounce
                            current_time = time.time()
                            if current_time - last_left_click_time > CLICK_DEBOUNCE_TIME:
                                try:
                                    pyautogui.click(button='left')
                                    last_left_click_time = current_time
                                    logging.info("Left mouse clicked.")
                                    print("Left mouse clicked.")
                                except Exception as e:
                                    logging.error(f"Error performing left mouse click: {e}")
                                    print(f"Error performing left mouse click: {e}")
                                # Visual Feedback (optional)
                                cv2.putText(frame, 'Left Click', (10, 120),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            pinch_count = 0

                # --- Visual Feedback for Pinch States (optional) ---
                if distance_thumb_index < THREE_FINGER_PINCH_THRESHOLD and distance_thumb_middle < THREE_FINGER_PINCH_THRESHOLD:
                    cv2.putText(frame, 'Three-Finger Pinch Detected', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Three-Finger Pinch: No', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if distance_thumb_index < PINCH_THRESHOLD:
                    cv2.putText(frame, 'Two-Finger Pinch Detected', (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Two-Finger Pinch: No', (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # If no hands are detected, reset the counters and release the mouse if holding
            if holding_left:
                try:
                    pyautogui.mouseUp(button='left')
                    holding_left = False
                    logging.info("Left mouse button released.")
                    print("Left mouse button released.")
                except Exception as e:
                    logging.error(f"Error releasing left mouse button: {e}")
                    print(f"Error releasing left mouse button: {e}")
            if holding_right:
                try:
                    pyautogui.mouseUp(button='right')
                    holding_right = False
                    logging.info("Right mouse button released.")
                    print("Right mouse button released.")
                except Exception as e:
                    logging.error(f"Error releasing right mouse button: {e}")
                    print(f"Error releasing right mouse button: {e}")
            pinch_count = 0
            three_pinch_count = 0
            prev_move_x, prev_move_y = 0, 0  # Reset movement deltas
            prev_index_x, prev_index_y = None, None  # Reset previous finger positions
            logging.info("No hand detected. Resetting states.")

        # --- Frame Rate Calculation ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Display FPS on frame (optional)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame with annotations
        cv2.imshow('Hand Gesture Control', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting application.")
            break

# -----------------------------
# Cleanup
# -----------------------------
video_capture.stop()
cv2.destroyAllWindows()
logging.info("Application closed.")
