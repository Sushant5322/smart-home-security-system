
import cv2
import numpy as np
from pathlib import Path
import time

import requests
import threading

UI_ENDPOINT = "http://127.0.0.1:5000/api/event"

def send_ui_event(name, confidence, snapshot_path=None):
    """Send event to UI in a non-blocking thread so recognizer loop won't hang."""
    payload = {
        "name": name,
        "confidence": float(confidence),
        "snapshot": snapshot_path or "/mnt/data/Screenshot 2025-11-03 at 10.56.00 PM.png"
    }
    def _post():
        try:
            requests.post(UI_ENDPOINT, json=payload, timeout=2)
        except Exception as e:
            print("[UI POST error]", e)
    threading.Thread(target=_post, daemon=True).start()

_last_ui_sent = {}        
_UI_DEBOUNCE_SECONDS = 5  

def process_recognition(name, confidence, frame=None):
    """
    Debounce + save snapshot + send UI event.
    Call this instead of printing or calling send_ui_event directly.
    `frame` should be the current BGR image from the camera loop (cv2 image).
    If frame is None, the function will still send event but without snapshot file.
    """
    import time
    now = time.time()
    last = _last_ui_sent.get(name, 0)
    if (now - last) < _UI_DEBOUNCE_SECONDS:
        
        return
    
    _last_ui_sent[name] = now

    
    snapshot_for_ui = "/static/default_snapshot.png"
    try:
        if frame is not None:
            snapshots_dir = os.path.join(os.path.dirname(__file__), "ui", "static", "snapshots")
            os.makedirs(snapshots_dir, exist_ok=True)
            snapshot_filename = f"{name}_{int(now)}.jpg"
            snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)
            snapshot_for_ui = f"/static/snapshots/{snapshot_filename}"
    except Exception as e:
        print("[Snapshot save error]", e)
        

    
    send_ui_event(name, confidence, snapshot_path=snapshot_for_ui)
    
    print(f"[ACTION] Recognized {name} — event sent to UI (confidence={confidence:.2f})")





model_path = Path("models/lbph_face_model.yml")
label_map_file = Path("models/labels.npy")


if not model_path.exists() or not label_map_file.exists():
    raise SystemExit("❌ Model or label map not found. Run train_recognizer.py first.")


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(model_path))
label_map = np.load(str(label_map_file), allow_pickle=True).item()
id_to_label = {v: k for k, v in label_map.items()}


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("🟢 Starting recognition — press 'q' to quit.")

last_seen = {}
DEBOUNCE = 5  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (200, 200))

        label_id, confidence = recognizer.predict(roi_resized)

        if confidence < 80:  
            name = id_to_label.get(label_id, "Unknown")
            
            text = f"{name} ({confidence:.1f})"
            process_recognition(name, confidence, frame)


            now = time.time()
            if name not in last_seen or (now - last_seen[name]) > DEBOUNCE:
                last_seen[name] = now
                print(f"[ACTION] Recognized {name} — perform automation here!")
        else:
            name = "Unknown"
            text = f"{name} ({confidence:.1f})"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Face Recognition (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
