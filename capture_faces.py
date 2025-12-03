import cv2
import os


dataset_path = "dataset_faces"
os.makedirs(dataset_path, exist_ok=True)


cam = cv2.VideoCapture(0)
cam.set(3, 640)  
cam.set(4, 480)  


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


face_id = input("\nEnter user name or ID: ")

print("\n[INFO] Initializing face capture. Look at the camera and wait ...")
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        
        file_name = os.path.join(dataset_path, f"{face_id}_{count}.jpg")
        cv2.imwrite(file_name, gray[y:y + h, x:x + w])

        cv2.imshow('Capturing Faces', img)

    
    key = cv2.waitKey(100) & 0xff
    if key == 27 or count >= 30:
        break

print("\n[INFO] Exiting and cleaning up...")
cam.release()
cv2.destroyAllWindows()
print(f"[INFO] {count} face images saved to '{dataset_path}'")
