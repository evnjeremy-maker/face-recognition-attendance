import cv2
import numpy as np
import xgboost as xgb
import pickle
import os
from datetime import datetime
from skimage.feature import local_binary_pattern

attendance_file = 'attendance.csv'
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write("ID,Name,Time,Date\n")


student_names = {
    1: "Student 1", 
    2: "Student 2",
    3: "Student 3"
}


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


model = xgb.XGBClassifier()
model.load_model('trainer/xgboost_face_model.json')

with open('trainer/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

radius = 1
n_points = 8 * radius
METHOD = 'uniform'

def get_lbph_features(image, grid_x=8, grid_y=8):
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    h, w = image.shape
    grid_h, grid_w = h // grid_y, w // grid_x
    histograms = []
    
    for i in range(grid_y):
        for j in range(grid_x):
            cell = lbp[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            hist, _ = np.histogram(cell.ravel(), bins=int(lbp.max() + 1), range=(0, int(lbp.max() + 1)))
            histograms.extend(hist)
            
    histograms = np.array(histograms, dtype=float)
    histograms /= (histograms.sum() + 1e-7)
    return histograms


logged_today = []

def mark_attendance(student_id, student_name):
    if student_id not in logged_today:
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        date_string = now.strftime('%Y-%m-%d')
        
        with open(attendance_file, 'a') as f:
            f.write(f"{student_id},{student_name},{time_string},{date_string}\n")
        
        logged_today.append(student_id)
        print(f"Logged attendance for {student_name} at {time_string}")

print("Starting camera for attendance... Press 'q' to quit.")
cam = cv2.VideoCapture(0)

while True:
    success, img = cam.read()
    if not success:
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
     
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (150, 150))
        
        
        features = get_lbph_features(face_resized)

        probabilities = model.predict_proba(features.reshape(1, -1))
        best_guess_index = np.argmax(probabilities)
        confidence = probabilities[0][best_guess_index] * 100

        if confidence > 60:
            predicted_encoded_label = best_guess_index
            predicted_id = encoder.inverse_transform([predicted_encoded_label])[0]
            name = student_names.get(predicted_id, "Unknown")
            

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{name} ({confidence:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            

            if name != "Unknown":
                mark_attendance(predicted_id, name)
        else:

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
    cv2.imshow('Attendance Scanner', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()