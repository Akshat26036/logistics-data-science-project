
# WEEK 4: LOGISTICS DELIVERY TIME PREDICTION

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

import joblib


# -----------------------------------
# 1. LOAD DATASET
# -----------------------------------

df = pd.read_csv("DataCo_Orders_Cleaned.csv")

print("Original Dataset Shape:", df.shape)


# -----------------------------------
# 2. DATE FEATURE ENGINEERING
# -----------------------------------

df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"]
)

df["Order_Year"] = df["order date (DateOrders)"].dt.year
df["Order_Month"] = df["order date (DateOrders)"].dt.month
df["Order_DayOfWeek"] = df["order date (DateOrders)"].dt.dayofweek


# -----------------------------------
# 3. SELECT FEATURES AND TARGET
# -----------------------------------

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


# -----------------------------------
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# -----------------------------------

# Sort records from oldest to newest
df = df.sort_values("order date (DateOrders)")

X = df[features]
y = df[target]

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# -----------------------------------
# 5. PREPROCESSING
# -----------------------------------

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

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# -----------------------------------
# 6. DEFINE MODELS
# -----------------------------------

models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}


# -----------------------------------
# 7. TRAIN AND EVALUATE MODELS
# -----------------------------------

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


# -----------------------------------
# 8. COMPARE MODEL PERFORMANCE
# -----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))


# -----------------------------------
# 9. SAVE RESULTS
# -----------------------------------

os.makedirs("week4_results", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "week4_results/model_comparison.csv",
    index=False
)

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/best_delivery_prediction_model.pkl"
)

print("\nBest Model:", best_model_name)

print("\nModel and results saved successfully!")
