import cv2
import face_recognition
import os
import pickle
import numpy as np
from datetime import datetime

DB_PATH = "face_data.pkl"
IMAGE_FOLDER = "faces"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_database(data):
    with open(DB_PATH, "wb") as f:
        pickle.dump(data, f)

def register_face(frame, face_encoding):
    name = input("New face detected. Enter name: ").strip()
    if name:
        data["encodings"].append(face_encoding)
        data["names"].append(name)
        filename = os.path.join(IMAGE_FOLDER, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] {name} registered successfully!")
        save_database(data)

def recognize_face(face_encoding):
    if not data["encodings"]:
        return "Unknown"

    matches = face_recognition.compare_faces(data["encodings"], face_encoding)
    face_distances = face_recognition.face_distance(data["encodings"], face_encoding)

    if matches and any(matches):
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return data["names"][best_match_index]
    return "Unknown"

data = load_database()
video = cv2.VideoCapture(0)

print("[INFO] Starting camera...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = recognize_face(face_encoding)

        if name == "Unknown":
            register_face(frame, face_encoding)
            name = data["names"][-1]  

        top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 5, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:  
        break

video.release()
cv2.destroyAllWindows()
