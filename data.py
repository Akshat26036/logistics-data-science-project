import pandas as pd
import numpy as np
df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin-1"
)

orders = df.drop_duplicates(subset="Order Id")

print("Total unique orders:", orders["Order Id"].nunique())

print("\n--- Delivery Status ---")
print(
    orders["Delivery Status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n--- Order Status ---")
print(
    orders["Order Status"]
    .value_counts(dropna=False)
    .to_string()
)

# Keep one record per order
orders = df.drop_duplicates(subset="Order Id").copy()

# Compare order status with delivery status
status_comparison = pd.crosstab(
    orders["Order Status"],
    orders["Delivery Status"],
    margins=True
)

print(status_comparison.to_string())

# Calculate percentage of each delivery status
delivery_percentages = (
    orders["Delivery Status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nDelivery status percentages:")
print(delivery_percentages.to_string())


# Check columns related to delivery performance
delivery_columns = [
    col for col in df.columns
    if any(word in col.lower() for word in [
        "delivery", "shipping", "shipment", "late_delivery_risk"
    ])
]

print("Delivery-related columns:")
for col in delivery_columns:
    print("-", col)

# Check whether actual and scheduled shipping days are available
for col in [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk"
]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).sort_index())

# Keep one row per unique order
orders = df.drop_duplicates(subset="Order Id").copy()

real = "Days for shipping (real)"
scheduled = "Days for shipment (scheduled)"

# Classify shipping performance
orders["Shipping Performance"] = "Unknown"

valid = (
    orders[real].notna()
    & orders[scheduled].notna()
)

orders.loc[
    valid & (orders[real] < orders[scheduled]),
    "Shipping Performance"
] = "Early"

orders.loc[
    valid & (orders[real] == orders[scheduled]),
    "Shipping Performance"
] = "On time"

orders.loc[
    valid & (orders[real] > orders[scheduled]),
    "Shipping Performance"
] = "Late"

# Keep canceled shipments separate
canceled = orders["Delivery Status"].eq("Shipping canceled")

orders.loc[canceled, "Shipping Performance"] = "Canceled"

# Display counts and percentages
counts = orders["Shipping Performance"].value_counts()

summary = pd.DataFrame({
    "Orders": counts,
    "Percentage": (counts / len(orders) * 100).round(2)
})

print("Total unique orders:", len(orders))
print("\nShipping performance:")
print(summary.to_string())

print("\nCompare calculated performance with original status:")
print(pd.crosstab(
    orders["Delivery Status"],
    orders["Shipping Performance"]
).to_string())

# One row per unique order
orders = df.drop_duplicates(subset="Order Id").copy()

# Calculate late-delivery rate by shipping mode
mode_analysis = (
    orders.groupby("Shipping Mode")
    .agg(
        Total_Orders=("Order Id", "nunique"),
        Late_Orders=(
            "Delivery Status",
            lambda s: (s == "Late delivery").sum()
        )
    )
)

mode_analysis["Late_Rate_Percent"] = (
    mode_analysis["Late_Orders"]
    / mode_analysis["Total_Orders"]
    * 100
).round(2)

mode_analysis = mode_analysis.sort_values(
    "Late_Rate_Percent",
    ascending=False
)

print(mode_analysis.to_string())
orders = df.drop_duplicates(subset="Order Id").copy()

# Calculate late delivery rate by market
market_analysis = (
    orders.groupby("Market")
    .agg(
        Total_Orders=("Order Id", "nunique"),
        Late_Orders=(
            "Delivery Status",
            lambda x: (x == "Late delivery").sum()
        )
    )
)

market_analysis["Late_Rate_Percent"] = (
    market_analysis["Late_Orders"]
    / market_analysis["Total_Orders"]
    * 100
).round(2)

print(market_analysis.sort_values(
    "Late_Rate_Percent", ascending=False
).to_string())

orders = df.drop_duplicates(subset="Order Id").copy()

region_analysis = (
    orders.groupby("Order Region")
    .agg(
        Total_Orders=("Order Id", "nunique"),
        Late_Orders=(
            "Delivery Status",
            lambda x: (x == "Late delivery").sum()
        )
    )
)

region_analysis["Late_Rate_Percent"] = (
    region_analysis["Late_Orders"]
    / region_analysis["Total_Orders"]
    * 100
).round(2)

print(
    region_analysis
    .sort_values("Late_Rate_Percent", ascending=False)
    .to_string()
)


# Rank regions by the actual number of late orders
region_priority = region_analysis.sort_values(
    "Late_Orders",
    ascending=False
)

print(
    region_priority[
        ["Total_Orders", "Late_Orders", "Late_Rate_Percent"]
    ].head(10).to_string()
)

orders = df.drop_duplicates(subset="Order Id").copy()

# Late-delivery rate for each region and shipping mode
region_shipping = (
    orders.groupby(["Order Region", "Shipping Mode"])
    .agg(
        Total_Orders=("Order Id", "nunique"),
        Late_Orders=(
            "Delivery Status",
            lambda x: (x == "Late delivery").sum()
        )
    )
)

region_shipping["Late_Rate_Percent"] = (
    region_shipping["Late_Orders"]
    / region_shipping["Total_Orders"]
    * 100
).round(2)

print(
    region_shipping
    .sort_values("Late_Rate_Percent", ascending=False)
    .head(15)
    .to_string()
)

orders = df.drop_duplicates(subset="Order Id").copy()

shipping_speed = (
    orders.groupby("Shipping Mode")
    .agg(
        Total_Orders=("Order Id", "nunique"),
        Avg_Actual_Days=("Days for shipping (real)", "mean"),
        Avg_Scheduled_Days=("Days for shipment (scheduled)", "mean"),
        Late_Orders=(
            "Delivery Status",
            lambda x: (x == "Late delivery").sum()
        )
    )
)

shipping_speed["Late_Rate_Percent"] = (
    shipping_speed["Late_Orders"]
    / shipping_speed["Total_Orders"]
    * 100
).round(2)

shipping_speed[
    ["Avg_Actual_Days", "Avg_Scheduled_Days",
     "Late_Rate_Percent"]
] = shipping_speed[
    ["Avg_Actual_Days", "Avg_Scheduled_Days",
     "Late_Rate_Percent"]
].round(2)

print(shipping_speed.sort_values("Avg_Actual_Days").to_string())

orders = df.drop_duplicates(subset="Order Id").copy()

real = "Days for shipping (real)"
scheduled = "Days for shipment (scheduled)"

# Calculate difference between actual and scheduled shipping days
orders["Delay_Days"] = orders[real] - orders[scheduled]

# Summarize shipping performance
delay_analysis = (
    orders.groupby("Shipping Mode")
    .agg(
        Total_Orders=("Order Id", "nunique"),
        Avg_Actual_Days=(real, "mean"),
        Avg_Scheduled_Days=(scheduled, "mean"),
        Avg_Delay_Days=("Delay_Days", "mean"),
        Max_Delay_Days=("Delay_Days", "max"),
        Late_Orders=(
            "Delivery Status",
            lambda x: (x == "Late delivery").sum()
        )
    )
)

delay_analysis["Late_Rate_Percent"] = (
    delay_analysis["Late_Orders"]
    / delay_analysis["Total_Orders"]
    * 100
)

print(delay_analysis.round(2).to_string())
import matplotlib.pyplot as plt

late_rates = delay_analysis["Late_Rate_Percent"].sort_values(
    ascending=False
)

late_rates.plot(kind="bar", figsize=(9, 5))

plt.title("Late Delivery Rate by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Late Delivery Rate (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()