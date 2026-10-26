import pandas as pd
import numpy as np
df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin-1"
)
print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nUnique orders:", df["Order Id"].nunique())
print("\nShipping mode counts:")
print(df["Shipping Mode"].value_counts(dropna=False))
# 1. Overall data audit
print("Dataset shape:", df.shape)
print("Exact duplicate rows:", df.duplicated().sum())
print("Total missing values:", df.isnull().sum().sum())

# 2. Missing values in all columns
missing = df.isnull().sum()
print("\nColumns with missing values:")
print(missing[missing > 0])

# 3. Verify order-level counts
orders = df.drop_duplicates(subset="Order Id").copy()

print("\nUnique orders:", orders["Order Id"].nunique())
print("\nShipping modes — unique orders:")
print(orders["Shipping Mode"].value_counts(dropna=False))

print("\nDelivery status — unique orders:")
print(orders["Delivery Status"].value_counts(dropna=False))

# 4. Check missing and duplicate order IDs
print("\nMissing Order IDs:", df["Order Id"].isnull().sum())
print("Unique shipping modes:", df["Shipping Mode"].nunique())

import pandas as pd
import numpy as np
df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin-1"
)


# Keep original data safe
df_clean = df.copy()

# Step 1: Remove columns that are completely empty
empty_cols = [
    col for col in df_clean.columns
    if df_clean[col].isna().all()
]

df_clean.drop(columns=empty_cols, inplace=True)

print("Removed fully empty columns:", empty_cols)

# Step 2: Convert date columns to datetime
date_columns = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for col in date_columns:
    df_clean[col] = pd.to_datetime(
        df_clean[col], errors="coerce"
    )

# Step 3: Recheck missing values
print("\nCleaned dataset shape:", df_clean.shape)
print("Exact duplicate rows:", df_clean.duplicated().sum())

print("\nRemaining missing values:")
print(
    df_clean.isnull().sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
)

# Step 4: Verify order count
print("\nUnique orders:", df_clean["Order Id"].nunique())
# 1. Check date columns
date_columns = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

print("\nDate validation:")
for col in date_columns:
    print(f"{col}:")
    print("  Missing / invalid dates:", df_clean[col].isna().sum())
    print("  Earliest date:", df_clean[col].min())
    print("  Latest date:", df_clean[col].max())

# 2. Check whether shipping dates precede order dates
order_date = df_clean["order date (DateOrders)"]
shipping_date = df_clean["shipping date (DateOrders)"]

invalid_sequence = (
    order_date.notna()
    & shipping_date.notna()
    & (shipping_date < order_date)
)

print("\nShipping dates before order dates:", invalid_sequence.sum())

# 3. Save the cleaned dataset
df_clean.to_csv("DataCoSupplyChain_Cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Columns expected to describe each order
order_columns = [
    "Order Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Shipping Mode",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Market",
    "Order Region",
    "Order Country",
    "Order Status",
    "Order Customer Id"
]

# Check whether any order has conflicting values
check_columns = order_columns[1:]

inconsistent = (
    df_clean.groupby("Order Id")[check_columns]
    .nunique(dropna=False)
    .gt(1)
)

print("Inconsistent order-level values by column:")
print(inconsistent.sum().sort_values(ascending=False))

print("\nOrders with at least one inconsistency:",
      inconsistent.any(axis=1).sum())
# Create an order-level dataset
order_columns = [
    "Order Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Shipping Mode",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Market",
    "Order Region",
    "Order Country",
    "Order Status",
    "Order Customer Id"
]

orders_df = (
    df_clean[order_columns]
    .drop_duplicates(subset="Order Id")
    .copy()
)

# Verify the result
print("\nOrder-level dataset shape:", orders_df.shape)
print("Unique Order IDs:", orders_df["Order Id"].nunique())
print("Duplicate Order IDs:", orders_df["Order Id"].duplicated().sum())

# Save order-level data
orders_df.to_csv("DataCo_Orders_Cleaned.csv", index=False)

print("\nOrder-level dataset saved successfully!")