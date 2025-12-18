import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# MLFLOW CONFIG 
# ============================

if os.getenv("GITHUB_ACTIONS"):
    DATA_PATH = "Sales-Transaction-v.4a_preprocessing.csv"  
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "Sales-Transaction-v.4a_preprocessing.csv")
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}")

mlflow.set_experiment("Sales Transaction - Linear Regression")
print("MLflow URI :", mlflow.get_tracking_uri())

# ============================
# LOAD DATA
# ============================
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

# ============================
# FEATURE & TARGET
# ============================
y = df["Price"]
X = df.drop(columns=["Price"])
X = X.astype("float64")
y = y.astype("float64")

# ============================
# SPLIT DATA
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# TRAIN & LOG MODEL
# ============================
with mlflow.start_run(run_name="Linear Regression - Sales Price"):
    mlflow.sklearn.autolog()  

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics manual
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log model secara eksplisit
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("\nTraining completed")
    print("MSE:", mse)
    print("R2 :", r2)