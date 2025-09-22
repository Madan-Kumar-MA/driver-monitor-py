import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib




df = pd.read_csv("phone_data.csv")

# Suppose first 1000 rows are focused, next 1000 rows are distracted
df['label'] = 0
df.loc[1000:2000, 'label'] = 1

df.to_csv("labeled_data.csv", index=False)
print("Labeled CSV saved!")
print(df.head())
print(df['label'].value_counts())
df = pd.read_csv("labeled_data.csv")
df = df.rename(columns={
    "Gyroscope x (rad/s)": "gx",
    "Gyroscope y (rad/s)": "gy",
    "Gyroscope z (rad/s)": "gz",
    "Absolute (rad/s)": "gyro_abs"
})
df = df.drop(columns=["Time (s)"])
print(df.head())
scaler = StandardScaler()
df[['gx','gy','gz','gyro_abs']] = scaler.fit_transform(df[['gx','gy','gz','gyro_abs']])
import numpy as np

# Create gyro magnitude (already have gyro_abs, but we can recompute if needed)
df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

print(df.head())
def create_features(data, window_size=50):
    features = []
    labels = []
    for i in range(0, len(data) - window_size, window_size):
        window = data.iloc[i:i+window_size]
        feat = {
            'gx_mean': window['gx'].mean(),
            'gy_mean': window['gy'].mean(),
            'gz_mean': window['gz'].mean(),
            'gyro_abs_mean': window['gyro_abs'].mean(),
            'gyro_mag_mean': window['gyro_mag'].mean(),
            'gx_std': window['gx'].std(),
            'gy_std': window['gy'].std(),
            'gz_std': window['gz'].std(),
            'gyro_abs_std': window['gyro_abs'].std(),
            'gyro_mag_std': window['gyro_mag'].std(),
        }
        features.append(feat)
        labels.append(window['label'].mode()[0])  # most frequent label in this window
    return pd.DataFrame(features), labels
X, y = create_features(df)
print(X.head())
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "driver_distraction_model.pkl")
print("Model saved as driver_distraction_model.pkl")

model = joblib.load("driver_distraction_model.pkl")
print("Model loaded!")
new_data = pd.DataFrame({
    "gx": np.random.normal(0, 0.2, 50),
    "gy": np.random.normal(0, 0.2, 50),
    "gz": np.random.normal(0, 0.2, 50),
    "gyro_abs": np.random.normal(0, 0.5, 50),
    "gyro_mag": np.random.normal(0, 0.5, 50)
})
def extract_features(window):
    feat = {
        'gx_mean': window['gx'].mean(),
        'gy_mean': window['gy'].mean(),
        'gz_mean': window['gz'].mean(),
        'gyro_abs_mean': window['gyro_abs'].mean(),
        'gyro_mag_mean': window['gyro_mag'].mean(),
        'gx_std': window['gx'].std(),
        'gy_std': window['gy'].std(),
        'gz_std': window['gz'].std(),
        'gyro_abs_std': window['gyro_abs'].std(),
        'gyro_mag_std': window['gyro_mag'].std(),
    }
    return pd.DataFrame([feat])

X_new = extract_features(new_data)
y_pred = model.predict(X_new)[0]

if y_pred == 0:
    print("✅ Driver is Focused")
else:
    print("⚠️ Driver is Distracted")





