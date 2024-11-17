import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load model
model = load_model('models/gesture_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.3 ,
                       max_num_hands=2)

labels = {0: "A", 1: 'B', 2: 'C'}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            feature = []
            for lm in hand_landmarks.landmark:
                feature.extend([lm.x, lm.y])

            # Predict gesture
            feature = np.expand_dims(feature, axis=0)
            prediction = np.argmax(model.predict(feature))
            label = labels[prediction]

            # Draw landmarks and label
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ISL Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()