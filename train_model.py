import cv2
import os
import numpy as np
import pickle
from facenet_helper import get_embedding

dataset_path = 'dataset'

embeddings=[]
labels = []

for image_name in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path,image_name)

    img = cv2.imread(img_path) 
    emb = get_embedding(img)
    id = int(image_name.split('.')[1])
    embeddings.append(emb)
    labels.append(id)

data = {
    "embeddings": embeddings,
    "labels": labels
}

with open("trainer/facenet_embeddings.pkl","wb") as f:
    pickle.dump(data,f)

print("FaceNet training completed")