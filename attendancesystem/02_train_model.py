import os
import cv2
import numpy as np
import xgboost as xgb
import pickle
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelEncoder


radius = 1
n_points = 8 * radius
METHOD = 'uniform'

def get_lbph_features(image, grid_x=8, grid_y=8):
    """
    Turns an image into a list of numbers by looking at the micro-textures 
    of the skin, breaking the face into an 8x8 grid.
    """
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

print("Reading faces from the dataset folder...")

image_paths = [os.path.join('dataset', f) for f in os.listdir('dataset') if f.endswith('.jpg')]
face_data = []
face_labels = []

for path in image_paths:

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (150, 150))

    user_id = int(os.path.split(path)[-1].split(".")[1])

    features = get_lbph_features(img)
    
    face_data.append(features)
    face_labels.append(user_id)

print(f"Extracted features from {len(face_data)} images. Training the XGBoost brain...")

X = np.array(face_data)
y = np.array(face_labels)


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)


model.fit(X, y_encoded)


model.save_model('trainer/xgboost_face_model.json')

with open('trainer/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("Success! Your model is trained and saved in the 'trainer' folder.")