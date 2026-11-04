
# WEEK 4: RANDOM FOREST TUNING AND VALIDATION

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# 1. LOAD DATA
df = pd.read_csv("DataCo_Orders_Cleaned.csv")

df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"]
)

# 2. CREATE DATE FEATURES
df["Order_Year"] = df["order date (DateOrders)"].dt.year
df["Order_Month"] = df["order date (DateOrders)"].dt.month
df["Order_DayOfWeek"] = df["order date (DateOrders)"].dt.dayofweek

# 3. SORT CHRONOLOGICALLY
df = df.sort_values("order date (DateOrders)").reset_index(drop=True)

features = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Market",
    "Order Region",
    "Order Country",
    "Order_Year",
    "Order_Month",
    "Order_DayOfWeek"
]

target = "Days for shipping (real)"

X = df[features]
y = df[target]

# 4. CHRONOLOGICAL 80:20 SPLIT
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. PREPROCESSING
categorical_features = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country"
]

numeric_features = [
    "Days for shipment (scheduled)",
    "Order_Year",
    "Order_Month",
    "Order_DayOfWeek"
]

preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

# 6. RANDOM FOREST PIPELINE
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ))
])

# 7. HYPERPARAMETER GRID
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 15],
    "model__min_samples_leaf": [2, 5]
}

# 8. TIME-SERIES CROSS-VALIDATION
tscv = TimeSeriesSplit(n_splits=3)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1,
    verbose=1
)

print("\nStarting hyperparameter tuning...")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation MAE:")
print(round(-search.best_score_, 4))

# 9. TEST SET EVALUATION
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nTUNED MODEL TEST RESULTS")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# 10. CREATE OUTPUT FOLDER
os.makedirs("week4_results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(
    best_model,
    "models/tuned_random_forest_model.pkl"
)

# Save metrics
metrics = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "CV MAE": -search.best_score_,
    "Test MAE": mae,
    "Test RMSE": rmse,
    "Test R2": r2
}])

metrics.to_csv(
    "week4_results/tuned_model_metrics.csv",
    index=False
)

# Save tuning results
pd.DataFrame(search.cv_results_).to_csv(
    "week4_results/hyperparameter_results.csv",
    index=False
)

# 11. ACTUAL VS PREDICTED GRAPH
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions,
    alpha=0.3
)

min_value = min(y_test.min(), predictions.min())
max_value = max(y_test.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Delivery Days")
plt.ylabel("Predicted Delivery Days")
plt.title("Actual vs Predicted Delivery Time")
plt.tight_layout()

plt.savefig(
    "week4_results/actual_vs_predicted.png",
    dpi=300
)

plt.close()

# 12. RESIDUAL PLOT
residuals = y_test - predictions

plt.figure(figsize=(8, 5))

plt.scatter(
    predictions,
    residuals,
    alpha=0.3
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Delivery Days")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.tight_layout()

plt.savefig(
    "week4_results/residual_plot.png",
    dpi=300
)

plt.close()

# 13. FEATURE IMPORTANCE
feature_names = (
    best_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = (
    best_model.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    "week4_results/feature_importance.csv",
    index=False
)

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.title("Top 15 Feature Importances")

plt.tight_layout()

plt.savefig(
    "week4_results/feature_importance.png",
    dpi=300
)

plt.close()

print("\nAll results and graphs saved successfully!")
