import os
import argparse
import sys
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# Argumen dari MLProject
# ======================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path", type=str, required=True,
    help="Path ke dataset CSV (absolut atau relatif terhadap repo)"
)
args = parser.parse_args()

# ======================
# Resolve path CSV
# ======================
if not os.path.isabs(args.input_path):
    # Asumsikan path relatif terhadap folder MLProject
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, args.input_path)
else:
    csv_path = args.input_path

if not os.path.isfile(csv_path):
    print(f"[ERROR] File tidak ditemukan: {csv_path}", file=sys.stderr)
    sys.exit(1)

# ======================
# MLflow setup
# ======================
# Hapus run lama jika ada
os.environ.pop("MLFLOW_RUN_ID", None)
mlflow.end_run(suppress=True)  # suppress=True agar tidak error kalau tidak ada run aktif

# Set tracking URI (gunakan SQLite)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set experiment (akan dibuat otomatis jika belum ada)
mlflow.set_experiment("Sales Transaction - Linear Regression")

# Pastikan folder mlruns ada (opsional, untuk run lokal)
os.makedirs("mlruns", exist_ok=True)

# ======================
# Load dataset
# ======================
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"[ERROR] Gagal membaca CSV: {e}", file=sys.stderr)
    sys.exit(1)

if "Price" not in df.columns:
    print("[ERROR] Kolom 'Price' tidak ditemukan di CSV", file=sys.stderr)
    sys.exit(1)

y = df["Price"].astype(float)
X = df.drop(columns=["Price"]).astype(float)

# ======================
# Split dataset
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Train & log model
# ======================
with mlflow.start_run(run_name="Linear Regression - Sales Price"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"R2 : {r2}")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_param("input_path", csv_path)

    mlflow.sklearn.log_model(model, artifact_path="model")
