import cv2
import mediapipe as mp
import pyautogui
import screeninfo
from selenium import webdriver
import keyboard

driver = webdriver.Chrome()
SECONDS_HOLD = 0.2

driver.get("https://poki.com/en/g/fruit-ninja")

driver.implicitly_wait(5)

monitor = screeninfo.get_monitors()
WINDOWS_WIDTH = monitor[0].width
WINDOWS_HEIGHT = monitor[0].height

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # # Draw the hand annotations on the image.
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initially set finger count to 0 for each cap
        fingerCount = 0

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Set variable to keep landmarks positions (x and y)
                handLandmarks = []

                # Fill list with x and y positions of each landmark
                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                mouse_y = handLandmarks[6][1]  # getting index finger lower part y_pos
                mouse_x = handLandmarks[6][0]  # getting index finger lower part x_pos
                to_pos_x = int(WINDOWS_WIDTH - mouse_x * WINDOWS_WIDTH)
                to_pos_y = int(mouse_y * WINDOWS_HEIGHT)

                # Other fingers: TIP y position must be lower than PIP y position,
                #   as image origin is in the upper left corner.
                if handLandmarks[8][1] < handLandmarks[6][1]:  # Index finger
                    fingerCount = fingerCount + 1
                    pyautogui.dragTo(to_pos_x, to_pos_y, SECONDS_HOLD, button='left')
                else:
                    pyautogui.moveTo(to_pos_x, to_pos_y)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display finger count
        # cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

        # Display image
        # cv2.imshow('MediaPipe Hands', image)
        if keyboard.is_pressed('q'):
            break

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
driver.quit()
