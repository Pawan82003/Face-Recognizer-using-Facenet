import numpy as np
import cv2
from keras_facenet import FaceNet

embedder = FaceNet()

def get_embedding(face_img):
    face_img = cv2.resize(face_img,(160,160))
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]
