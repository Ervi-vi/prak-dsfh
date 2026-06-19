import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Thyroid_Diff.csv")

# Encode semua kolom kategorikal
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Pisahkan fitur dan target
X = df.drop("Recurred", axis=1)
y = df["Recurred"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# Simpan model
joblib.dump(model, "model.pkl")

# Simpan encoder
joblib.dump(encoders, "encoders.pkl")

print("Model berhasil disimpan")
