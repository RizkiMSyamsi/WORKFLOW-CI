import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset di folder yang sama
DATA_PATH = "Sales-Transaction-v.4a_preprocessing.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

y = df["Price"]
X = df.drop(columns=["Price"]).select_dtypes(include=["float64", "int64"]).astype("float64")
y = y.astype("float64")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



mlflow.sklearn.autolog(log_models=False) 

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics manual
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)

# Log model
mlflow.sklearn.log_model(model, artifact_path="model")

# Ambil run_id dari active run
active_run = mlflow.active_run()
if active_run:
    print(f"\nTraining completed successfully!")
    print(f"Run ID: {active_run.info.run_id}")
    print("MSE:", mse)
    print("R2 :", r2)
else:
    print("No active run detected")