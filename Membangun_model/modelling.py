
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset hasil preprocessing
df = pd.read_csv('dataset_preprocessing/personality_dataset_preprocessing.csv')

# 2. Pisahkan fitur dan target
X = df.drop('Personality', axis=1)
y = df['Personality']

# 3. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# 5. Inisialisasi MLflow Tracking (opsional jika sudah diatur default)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Personality Classification")

# 6. Jalankan pelatihan model dalam MLflow run
with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 7. Evaluasi dan prediksi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Akurasi Model: {acc:.4f}")
