
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
df = pd.read_csv("DataCo_Orders_Cleaned.csv")

# 2. Convert date and extract date features
df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"]
)

df["Order_Year"] = df["order date (DateOrders)"].dt.year
df["Order_Month"] = df["order date (DateOrders)"].dt.month
df["Order_DayOfWeek"] = df["order date (DateOrders)"].dt.dayofweek

# 3. Sort chronologically
df = df.sort_values("order date (DateOrders)").reset_index(drop=True)

# 4. Define target
target = "Days for shipping (real)"
y = df[target]

# 5. Define two feature sets
full_features = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Market",
    "Order Region",
    "Order Country",
    "Order_Year",
    "Order_Month",
    "Order_DayOfWeek"
]

reduced_features = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Order_Year",
    "Order_Month",
    "Order_DayOfWeek"
]

# 6. Chronological 80/20 split
split_index = int(len(df) * 0.80)

X_train_full = df[full_features].iloc[:split_index]
X_test_full = df[full_features].iloc[split_index:]

X_train_reduced = df[reduced_features].iloc[:split_index]
X_test_reduced = df[reduced_features].iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 7. Function to train and evaluate a model
def train_evaluate(X_train, X_test, y_train, y_test, model_name):

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }


# 8. Train both models
results = []

results.append(
    train_evaluate(
        X_train_full, X_test_full,
        y_train, y_test,
        "With Scheduled Days"
    )
)

results.append(
    train_evaluate(
        X_train_reduced, X_test_reduced,
        y_train, y_test,
        "Without Scheduled Days"
    )
)

# 9. Compare results
comparison = pd.DataFrame(results)

print("\nFINAL FEATURE COMPARISON")
print(comparison.to_string(index=False))

# 10. Save results
import os

os.makedirs("week4_results", exist_ok=True)

comparison.to_csv(
    "week4_results/feature_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
