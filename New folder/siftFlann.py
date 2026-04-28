import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# getcwd -> buat .ipynb
# __file__ -> buat .py

try:
    BASE_DIR = os.path.dirname(__file__)
except:
    BASE_DIR = os.getcwd()

dataset_dir = os.path.join(BASE_DIR, 'source')
target_dir = os.path.join(BASE_DIR, 'target')

dataset_images = []
for img in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, img)
    dataset_images.append(path)

target_images = []
for img in os.listdir(target_dir):
    path = os.path.join(target_dir, img)
    target_images.append(path)

# CARA SINGKAT
# dataset_images = [os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir)]
# target_images = [os.path.join(target_dir, img) for img in os.listdir(target_dir)]

target_img = cv2.imread(target_images[0])

target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
target_gray = cv2.GaussianBlur(target_gray, ksize=(3,3), sigmaX=0)

sift = cv2.SIFT_create()

target_sift_kp, target_sift_desc = sift.detectAndCompute(target_gray, None)
target_sift_desc = np.float32(target_sift_desc)

best_match = [0, None, None, None, None]
# number_of_valid_matches, image, kp, matches, matches_mask

for dataset_img in dataset_images:
    obj = cv2.imread(dataset_img)
    obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    obj_gray = cv2.medianBlur(obj_gray, 3)

    obj_sift_kp, obj_sift_desc = sift.detectAndCompute(obj_gray, None)
    obj_sift_desc = np.float32(obj_sift_desc)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1), dict(checks=50))
    # algorithm=1 -> KDTree
    # checks=50 -> jmlh cek yg dilakukan algorithm
    # semakin tinggi, semakin akurat

    matches = flann.knnMatch(target_sift_desc, obj_sift_desc, 2)
    # k = 2 -> Lowe's Ratio Test -> nearest neighbor 2

    matches_mask = [[0,0] for _ in range(len(matches))]

    current_matches = 0

    for i, (first, second) in enumerate(matches):
        # Lowe's -> d1 < 0.7 * d2
        if first.distance < 0.7 * second.distance:
            matches_mask[i] = [1,0]
            current_matches += 1
    
    if current_matches > best_match[0]:
        best_match = (current_matches, obj, obj_sift_kp, matches, matches_mask)

max_matches, obj, obj_sift_kp, matches, matches_mask = best_match

sift_res = cv2.drawMatchesKnn(
    target_img, target_sift_kp,
    obj, obj_sift_kp,
    matches,
    None, 
    [0, 0, 255],
    [0, 255, 0],
    matches_mask
)

rgb_res = cv2.cvtColor(sift_res, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.title("Best Match Result")
plt.imshow(rgb_res)
plt.axis(False)
plt.show()
