import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 30.0

print("Recording FPS:", fps)

out = cv2.VideoWriter("gesture_recording.mp4", fourcc, fps, (frame_width, frame_height))


prev_x, prev_y = None, None
last_action_time = 0
cooldown = 0.4 

last_fist_time = 0
fist_cooldown = 2.0  


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

            lm = handLms.landmark[0]
            cx, cy = int(lm.x * w), int(lm.y * h)

            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            now = time.time()

            tips_ids = [8, 12, 16, 20]
            base_ids = [5, 9, 13, 17]

            fingers_folded = 0
            for tip_id, base_id in zip(tips_ids, base_ids):
                tip = handLms.landmark[tip_id]
                base = handLms.landmark[base_id]
                if tip.y > base.y:
                    fingers_folded += 1

            if fingers_folded >= 4 and (now - last_fist_time) > fist_cooldown:
                print("FIST DETECTED -> HOVERBOARD!")
                action_text = "HOVERBOARD!"

                pyautogui.press("up")
                time.sleep(0.05)
                pyautogui.press("up")

                last_fist_time = now

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

    cv2.imshow("Gesture Control", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as gesture_recording.mp4")
