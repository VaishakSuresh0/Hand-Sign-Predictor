import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    min_val = np.min(landmarks)
    max_val = np.max(landmarks)
    return (landmarks - min_val) / (max_val - min_val)

# Collect Training Data
cap = cv2.VideoCapture(0)
data = []

for number in range(10):
    print(f"Capturing 1000 samples for number {number}. Press 'q' to quit.")
    captured = 0
    while captured < 1000:
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
                landmarks = normalize_landmarks(landmarks)
                data.append([landmarks, number])
                captured += 1
                print(f"Captured {captured}/1000 samples for {number}.")

        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Convert data to numpy array
np.save("sign_language_data.npy", np.array(data, dtype=object))
print("Data collection complete and saved!")

# Load the dataset
data = np.load("sign_language_data.npy", allow_pickle=True)
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# Normalize the data
X = np.array(X).reshape(-1, 42, 1)  # Reshaping for CNN
y = to_categorical(y, num_classes=10)  # One-hot encoding labels

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(42, 1)),
    BatchNormalization(),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16)

# Save the new model
model.save("train_sign_language.h5")
print("New model trained and saved successfully!")
