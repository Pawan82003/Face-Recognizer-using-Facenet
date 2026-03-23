import cv2
from facenet_helper import get_embedding
import pickle
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open("trainer/facenet_embeddings.pkl","rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
labels = data["labels"]

names = {1:"Pawan",2:"Amit"}

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.2,5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h,x:x+w] 
        emb = get_embedding(face) 
        distances = np.linalg.norm(known_embeddings - emb, axis=1) 
        min_index = np.argmin(distances) 
        if distances[min_index] < 0.9: 
            name = names[labels[min_index]] 
        else: 
            name = "Unknown"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow('Face Recognition',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()