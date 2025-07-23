import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model("train_sign_language.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    min_val = np.min(landmarks)
    max_val = np.max(landmarks)
    return (landmarks - min_val) / (max_val - min_val)

# Start Webcam
cap = cv2.VideoCapture(0)
print("Show a hand sign to predict.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
            landmarks = normalize_landmarks(landmarks).reshape(1, 42, 1)  # Correct input shape for CNN

            # Predict the number using the trained model
            predictions = model.predict(landmarks)
            predicted_number = np.argmax(predictions)
            confidence = np.max(predictions) * 100  # Convert to percentage

            # Determine if the number is odd or even
            odd_even = "Even" if predicted_number % 2 == 0 else "Odd"

            # Display prediction, confidence, and odd/even status
            cv2.putText(frame, f"Predicted: {predicted_number} ({confidence:.2f}%)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Type: {odd_even}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
