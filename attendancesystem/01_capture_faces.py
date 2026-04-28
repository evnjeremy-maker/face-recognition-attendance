import cv2
import os


if not os.path.exists('dataset'):
    os.makedirs('dataset')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


face_id = input("Enter a numeric ID for this student (e.g., 1 for John) and press Enter: ")
print("\nStarting the camera. Look at the lens and move your head slightly...")


cam = cv2.VideoCapture(0)
count = 0

while True:
  
    success, img = cam.read()
    
    if not success:
        print("Failed to grab camera frame. Check your webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
        

        cv2.imshow('Face Registration - Press Q to cancel', img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count >= 50:
        break


print(f"\nSuccessfully saved {count} face images to the dataset folder!")
cam.release()
cv2.destroyAllWindows()