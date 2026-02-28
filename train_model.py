import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "dataset"

X = []
y = []

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

    return np.hstack([mfcc, chroma, mel])


for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            # Emotion code is 3rd value in filename
            emotion_code = int(file.split("-")[2])

            # Map emotions to stress levels
            if emotion_code in [1, 2]:      # Neutral, Calm
                label = 0  # Low Stress
            elif emotion_code == 3:         # Happy
                label = 1  # Medium Stress
            elif emotion_code in [5, 6]:    # Angry, Fearful
                label = 2  # High Stress
            else:
                continue  # Skip other emotions

            features = extract_features(file_path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature extraction completed.")
print("Total samples:", len(X))

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully.")