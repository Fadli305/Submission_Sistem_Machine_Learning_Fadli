import dagshub
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dagshub import dagshub_logger

# Inisialisasi koneksi ke DagsHub

dagshub.init(repo_owner='anakxsat',
             repo_name='Submission_Sistem_Machine_Learning_Fadli',
             mlflow=True)

# Load dataset

df = pd.read_csv('dataset_preprocessing/personality_dataset_preprocessing.csv')
X = df.drop('Personality', axis=1)
y = df['Personality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment

mlflow.set_experiment("Personality Classification - Tuning")

# Define models and param grids

models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 10]
        }
    }
}

# Loop over models

for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"\n🔍 Tuning {model_name}...")

        # Grid search
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Best model & params
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Cross-validation score
        cv_acc = cross_val_score(best_model, X_train, y_train, cv=3, scoring='accuracy').mean()
        mlflow.log_metric("cv_accuracy", cv_acc)

        # Confusion matrix visualization
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        os.makedirs(f"saved_models/{model_name}", exist_ok=True)
        cm_path = f"saved_models/{model_name}/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Logging params & metrics to MLflow
        mlflow.log_param("model", model_name)
        for param_name, value in grid.best_params_.items():
            mlflow.log_param(param_name, value)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Save and log model
        joblib.dump(best_model, f"saved_models/{model_name}/model.pkl")
        mlflow.log_artifact(f"saved_models/{model_name}/model.pkl", artifact_path=f"best_model_{model_name}")

        # Log confusion matrix plot
        mlflow.log_artifact(cm_path, artifact_path=f"best_model_{model_name}")

        print(f"✅ {model_name} done - Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, CV Accuracy: {cv_acc:.4f}")

        # Link Dagshub "https://dagshub.com/anakxsat/Submission_Sistem_Machine_Learning_Fadli.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D"