import cv2
from deepface import DeepFace

cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

while True:
    ret,frame=cap.read()

    if not ret:
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]
    try:
        result = DeepFace.analyze=['emotion'], enforce_detection=False
        emotion = result[0]['dominant_emotion']
    except:
        emotion = "unknown"

    cv2.