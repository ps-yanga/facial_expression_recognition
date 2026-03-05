import cv2
from deepface import DeepFace
import time

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

min_face_size = 50
emotion_cache = {}
cache_duration = 0.5

last_update_time = 0

while True:
    ret,frame = cap.read()
    if not ret:
        break

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        if w < min_face_size or h < min_face_size:
            continue
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        face_roi = frame[y:y+h,x:x+w]
        if face_roi.size == 0:
            print("No face detected")
            continue
        current_time = time.time()
        face_key = f"{x},{y},{w},{h}"

        if face_key in emotion_cache:
            last_emotion_time = emotion_cache[face_key]
            if current_time - last_emotion_time < cache_duration:
                emotion=emotion_cache[face_key]
            else:
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                    emotion = result[0]['dominant_emotion']
                    emotion_cache[face_key] = emotion
            except Exception as e:
                print(f"DeepFace Error: {e}")
                emotion = "unknown"
                emotion_cache[face_key] = emotion
        else:
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                emotion = result[0]['dominant_emotion']
                print("Emotion Detected:,{emotion}")
            except Exception as e:
                print(f"DeepFace Error: {e}")
                emotion = "unknown"

            cv2.rectangle(frame,(x, y),(x + w, y + h), (0, 255, 0), 2)

            text_y = y - 10 if y > 10 else y + h + 10
            cv2.putText(frame, emotion, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 2)

    cv2.imshow('Facial Emotion Recognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()