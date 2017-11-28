import sys
import os
import cv2
import numpy as np
from label_faces import load_csv

def create_recognizer(faces_dir, labels_file_name):
    labels_dict = load_csv(labels_file_name)
    faces = []
    labels = []

    all_files = [f for f in os.listdir(faces_dir) if os.path.isfile(os.path.join(faces_dir, f))]
    for fn in all_files:
        img = cv2.imread(os.path.join(faces_dir, fn))
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces.append(grey)
        if labels_dict[fn] == "True":
            labels.append(1)
        else:
            labels.append(0)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    return face_recognizer

# TODO: Do stuff with this recognizer

if __name__ == "__main__":
    faces_dir = sys.argv[1]
    labels_fn = sys.argv[2]
    recognizer = create_recognizer(faces_dir, labels_fn)
