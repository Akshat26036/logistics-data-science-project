
import pandas as pd

data = pd.read_csv(
    r"C:\Users\think\OneDrive\Desktop\week1 project\DataCoSupplyChainDataset.csv",
    encoding="latin1"
)


for i, column in enumerate(data.columns, start=1):
    print(f"{i}. {column}")

# Order dates ko datetime format mein convert karo
data["order date (DateOrders)"] = pd.to_datetime(
    data["order date (DateOrders)"],
    errors="coerce"
)

# Total unique orders
total_orders = data["Order Id"].nunique()

# First aur last order date
first_date = data["order date (DateOrders)"].min()
last_date = data["order date (DateOrders)"].max()

print("Total Unique Orders:", total_orders)
print("First Order Date:", first_date)
print("Last Order Date:", last_date)

data["order date (DateOrders)"] = pd.to_datetime(
    data["order date (DateOrders)"],
    errors="coerce"
)

# Extract year from order date
data["Year"] = data["order date (DateOrders)"].dt.year

# Count unique orders for each year
yearly_orders = (
    data.groupby("Year")["Order Id"]
    .nunique()
    .reset_index(name="Total Orders")
)

print(yearly_orders.to_string(index=False))

# Extract year and month from order date
data["YearMonth"] = (
    data["order date (DateOrders)"].dt.to_period("M")
)

# Count unique orders per month
monthly_orders = (
    data.groupby("YearMonth")["Order Id"]
    .nunique()
    .reset_index(name="Total Orders")
)

print(monthly_orders.to_string(index=False))

# Find the month with the highest number of orders
highest_month = monthly_orders.loc[
    monthly_orders["Total Orders"].idxmax()
]

print("\nMonth with highest orders:")
print(highest_month)

import matplotlib.pyplot as plt

# Convert YearMonth into a string for plotting
monthly_orders["Month"] = (
    monthly_orders["YearMonth"].astype(str)
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_orders["Month"],
    monthly_orders["Total Orders"],
    marker="o"
)

plt.title("Monthly Order Trend (2015–2018)")
plt.xlabel("Year-Month")
plt.ylabel("Number of Unique Orders")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
