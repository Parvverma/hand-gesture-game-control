import cv2
import mediapipe as mp
import pyautogui
import time
import math

# --------- Setup ---------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Get frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --------- Video Writer Setup ---------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30.0
print("Recording FPS:", fps)

out = cv2.VideoWriter("gesture_recording.mp4", fourcc, fps, (frame_width, frame_height))

# --------- Timing ---------
prev_x, prev_y = None, None
last_action_time = 0
cooldown = 0.4  # seconds between movement key presses

last_fist_time = 0
fist_cooldown = 2.0  # seconds between hoverboard activations

# --------- Helper: count open fingers ---------
# Landmark indices:
# Thumb tip: 4, Index tip: 8, Middle: 12, Ring: 16, Pinky: 20
# Bases: Index base: 5, Middle: 9, Ring: 13, Pinky: 17
def count_fingers(handLms):
    fingers = 0

    # For simplicity, ignore thumb (it’s trickier with orientation)
    tips = [8, 12, 16, 20]
    bases = [5, 9, 13, 17]

    for tip_id, base_id in zip(tips, bases):
        tip = handLms.landmark[tip_id]
        base = handLms.landmark[base_id]
        # If tip is above base in image, finger is open
        if tip.y < base.y:
            fingers += 1

    return fingers

# --------- Loop ---------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    action_text = "No hand"

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            now = time.time()

            # --------- Get wrist for movement ---------
            wrist = handLms.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # --------- Count fingers ---------
            fingers_open = count_fingers(handLms)
            cv2.putText(frame, f"Fingers: {fingers_open}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # --------- Fist = Hoverboard (0 fingers) ---------
            if fingers_open == 0 and (now - last_fist_time) > fist_cooldown:
                print("FIST (0 fingers) -> HOVERBOARD!")
                action_text = "HOVERBOARD!"

                # Double tap UP arrow
                pyautogui.press("space")

                last_fist_time = now

            # --------- Movement Control (only if not just triggered) ---------
            if prev_x is not None and prev_y is not None:
                dx = cx - prev_x
                dy = cy - prev_y

                if now - last_action_time > cooldown:
                    if dx > 40:
                        pyautogui.press("right")
                        action_text = "RIGHT"
                        last_action_time = now
                    elif dx < -40:
                        pyautogui.press("left")
                        action_text = "LEFT"
                        last_action_time = now
                    elif dy < -40:
                        pyautogui.press("up")
                        action_text = "UP (JUMP)"
                        last_action_time = now
                    elif dy > 40:
                        pyautogui.press("down")
                        action_text = "DOWN (SLIDE)"
                        last_action_time = now

            prev_x, prev_y = cx, cy

    cv2.putText(frame, f"Action: {action_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show window
    cv2.imshow("Gesture Control", frame)

    # Save frame to video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# --------- Cleanup ---------
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as gesture_recording.mp4")
