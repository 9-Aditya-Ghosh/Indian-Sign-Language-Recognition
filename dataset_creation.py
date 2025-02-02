import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
features, labels = [], []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    for img_path in os.listdir(class_dir):
        img = cv2.imread(os.path.join(class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                feature = []
                for lm in hand_landmarks.landmark:
                    feature.extend([lm.x, lm.y])
                features.append(feature)
                labels.append(int(dir_))

# Save dataset as pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': features, 'labels': labels}, f)