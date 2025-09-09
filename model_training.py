# model_training.py
import os, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAINING_CSV = os.path.join(DATA_DIR, "dosage_training.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dosage_model.joblib")

if not os.path.exists(TRAINING_CSV):
    raise FileNotFoundError(f"Put training CSV at {TRAINING_CSV}")

df = pd.read_csv(TRAINING_CSV)
df['drug'] = df['drug'].astype(str).str.lower().str.strip()
le = LabelEncoder()
df['drug_encoded'] = le.fit_transform(df['drug'])

X = df[['drug_encoded','age','dose']].astype(float)
y = df['safe'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": clf, "label_encoder": le}, f)
print("Saved model to", MODEL_PATH)
