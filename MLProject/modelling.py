import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

if __name__ == "__main__":
    # Ambil parameter jika ada (untuk future flexibility, meski LinearRegression tidak punya hyperparam utama)
    # Contoh: bisa tambah parameter lain nanti
    file_path = sys.argv[1] if len(sys.argv) > 1 else "Sales-Transaction-v.4a_preprocessing.csv"

    # Path relatif aman untuk MLflow Project
    data = pd.read_csv(file_path)

    y = data["Price"]
    X = data.drop(columns=["Price"])

    X = X.astype("float64")
    y = y.astype("float64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = X_train[:5]  # Untuk log model

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metric (manual, lebih kontrol)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print("\nTraining completed")
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2 :", r2)