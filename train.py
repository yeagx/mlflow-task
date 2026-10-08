import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Part 1: load and split
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
print(X.head(3))  # sample of the data
print(y.head(3))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Part 2 and 3: three runs
mlflow.set_experiment("California Housing")

configs = [
    {"max_depth": 3, "learning_rate": 0.1},
    {"max_depth": 5, "learning_rate": 0.05},
    {"max_depth": 7, "learning_rate": 0.01},
]

for i, cfg in enumerate(configs, start=1):
    with mlflow.start_run(run_name=f"Run {i}"):
        model = GradientBoostingRegressor(random_state=42, **cfg)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        mlflow.log_params(cfg)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"Run {i} | {cfg} | RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

# Part 4: best run by RMSE
best = mlflow.search_runs(order_by=["metrics.rmse ASC"]).iloc[0]
print(f"\nBest: {best['tags.mlflow.runName']} | RMSE={best['metrics.rmse']:.4f}")
