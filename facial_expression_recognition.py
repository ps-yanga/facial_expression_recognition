import cv2
from deepface import DeepFace

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret,frame=cap.read()
    if not ret:
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face_roi = frame[y:y+h,x:x+w]

    try:
        result = DeepFace.analyze(face_roi,action=['emotion'], enforce_detection=False, silent=True)
        emotion = result[0]['dominant_emotion']
    except:
        emotion = "unknown"

    cv2.rectangle(frame,(x, y),(x + w, y + h), (0, 255, 0), 2)

    text_y = y - 10 if y > 10 else y + h + 10
    cv2.putText(frame, emotion, (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0,0,255), 2)

    cv2.imshow('Facial Emotion Recognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()