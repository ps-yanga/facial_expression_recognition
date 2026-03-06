import cv2
from deepface import DeepFace
import time
from collections import Counter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

min_face_size = 80
emotion_history = {}
history_length = 15
update_interval = 5.0
face_tracking = {}

while True:
    ret,frame = cap.read()
    if not ret:
        break

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    current_faces = []

    for (x,y,w,h) in faces:
        if w < min_face_size or h < min_face_size:
            continue

        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        face_roi = frame[y1:y2, x1:x2]

        if face_roi.size == 0:
            print("No face detected")
            continue

        face_center = (x + w // 2, y + h // 2)
        face_key = f"{face_center[0]:.0f},{face_center[1]:.0f}"


        if face_key not in face_tracking:
            face_tracking[face_key] = {'emotions': [], 'last_update': 0, 'last_rect': (x, y, w, h) }
            last_update = face_tracking[face_key]['last_update']

            if time.time() - last_update < update_interval:
                emotions = face_tracking[face_key]['emotions']
                if emotions:
                    emotion = Counter(emotions).most_common(1)[0][0]
                else:
                    emotion = "neutral"
            else:
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                    emotion = result[0]['dominant_emotion']

                    face_tracking[face_key]['emotions'].append(emotion)
                    face_tracking[face_key]['emotions'] = face_tracking[face_key]['emotions'][-history_length:]
                    face_tracking[face_key]['last_update'] = time.time()
                except Exception as e:
                    print(f"DeepFace Error: {e}")
                    emotion = "unknown"

            tracked_rect = face_tracking[face_key]['last_rect']
            new_rect = (x, y, w, h)
            smoothed_rect = (
                int(tracked_rect[0] * 0.7 + new_rect[0] * 0.3),
                int(tracked_rect[1] * 0.7 + new_rect[1] * 0.3),
                int(tracked_rect[2] * 0.7 + new_rect[2] * 0.3),
                int(tracked_rect[3] * 0.7 + new_rect[3] * 0.3)
            )
            cv2.rectangle(frame,(smoothed_rect[0], smoothed_rect[1]),(smoothed_rect[0] + smoothed_rect[2], smoothed_rect[1] + smoothed_rect[3]), (0, 255, 0), 2)

            face_tracking[face_key]['last_rect'] = smoothed_rect

            text_y = y - 10 if y > 10 else y + h + 10
            cv2.putText(frame, emotion, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 2)

            current_faces.append(face_key)

    key_to_remove = [key for key in face_tracking if key not in current_faces]
    for key in key_to_remove:
        del face_tracking[key]

    cv2.imshow('Facial Emotion Recognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(0.05)
cap.release()
cv2.destroyAllWindows()