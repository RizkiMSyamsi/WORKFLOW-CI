import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset harus ada di root repository
DATA_PATH = "Sales-Transaction-v.4a_preprocessing.csv"

mlflow.set_experiment("Sales Transaction - Linear Regression")

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

y = df["Price"]
X = df.drop(columns=["Price"]).astype("float64")
y = y.astype("float64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="Linear Regression - Sales Price") as run:
    mlflow.sklearn.autolog(log_models=False)  # autolog tapi model log manual

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Penting: log model secara eksplisit agar bisa di-build docker
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Run ID: {run.info.run_id}")
    print("MSE:", mse)
    print("R2:", r2)