# train_recognizer.py
import cv2
import os
import numpy as np
from pathlib import Path

dataset_path = Path("dataset_faces")
model_path = Path("models")
model_path.mkdir(exist_ok=True, parents=True)
recognizer_file = model_path / "lbph_face_model.yml"
label_map_file = model_path / "labels.npy"

faces = []
labels = []
label_to_id = {}
next_id = 0

# iterate through images
for img_file in sorted(dataset_path.glob("*.jpg")):
    name = img_file.stem.split("_")[0]  # expects NAME_index.jpg format
    if name not in label_to_id:
        label_to_id[name] = next_id
        next_id += 1
    id_ = label_to_id[name]

    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.resize(img, (200, 200))
    faces.append(img)
    labels.append(id_)

if not faces:
    raise SystemExit("❌ No face images found in dataset_faces/. Run capture_faces.py first!")

faces_np = np.array(faces)
labels_np = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces_np, labels_np)
recognizer.write(str(recognizer_file))
np.save(str(label_map_file), label_to_id)

print(f"✅ Trained model saved to {recognizer_file}")
print(f"✅ Label map saved to {label_map_file}")
print("Labels:", label_to_id)
