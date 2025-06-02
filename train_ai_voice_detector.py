import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Constants
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
N_MFCC = 20

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio = librosa.effects.preemphasis(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = librosa.util.fix_length(mfcc, size=63, axis=1)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc.flatten()

X, y = [], []

for label, category in enumerate(["human", "ai"]):
    folder = os.path.join(DATASET_PATH, category)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            features = extract_features(path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/ai_voice_detector_model.pkl")
print("✅ Model saved to models/ai_voice_detector_model.pkl")
