import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

face_id = input("Enter user ID: ")

count = 0

while True:

    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        count += 1

        cv2.imwrite("dataset/User."+str(face_id)+"."+str(count)+".jpg",
                    gray[y:y+h,x:x+w])

    cv2.imshow('Capturing Faces',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= 50:
        break

cap.release()
cv2.destroyAllWindows()