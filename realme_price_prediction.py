
# =========================================================
# REALME SMARTPHONE PRICE PREDICTION
# Machine Learning Project
# Algorithm: Random Forest Regressor
# =========================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. Create Realme Smartphone Dataset
# ---------------------------------------------------------

data = {
    "RAM": [3, 4, 4, 6, 6, 8, 8, 8, 12, 12, 16, 4, 6, 8, 12],
    "Storage": [32, 64, 128, 128, 256, 128, 256, 256, 256, 512, 512, 128, 128, 256, 512],
    "Battery": [5000, 5000, 5000, 5000, 5000, 5000, 5000, 6000, 5000, 5000, 5000, 5000, 5000, 6000, 5000],
    "Camera": [13, 50, 50, 64, 64, 50, 108, 50, 50, 200, 50, 50, 64, 108, 200],
    "5G": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "Price": [7999, 9999, 12999, 14999, 17999, 19999, 24999,
              22999, 29999, 39999, 49999, 10999, 15999, 26999, 59999]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2. Select Input and Output
# ---------------------------------------------------------

X = df[["RAM", "Storage", "Battery", "Camera", "5G"]]
y = df["Price"]

# ---------------------------------------------------------
# 3. Split Dataset into Training and Testing
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. Train Random Forest Model
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Evaluate Model
# ---------------------------------------------------------

y_pred = model.predict(X_test)

print("\n===== REALME PRICE PREDICTION =====")
print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, y_pred), 2))

print("R2 Score:",
      round(r2_score(y_test, y_pred), 2))

# ---------------------------------------------------------
# 6. Predict Price for a New Realme Phone
# ---------------------------------------------------------

new_phone = pd.DataFrame([{
    "RAM": 8,
    "Storage": 256,
    "Battery": 5000,
    "Camera": 50,
    "5G": 1
}])

predicted_price = model.predict(new_phone)[0]

print("\nNew Phone Specifications")
print("Brand: Realme")
print("RAM: 8 GB")
print("Storage: 256 GB")
print("Battery: 5000 mAh")
print("Camera: 50 MP")
print("5G: Yes")

print("\nPredicted Price: ₹",
      round(predicted_price))

# ---------------------------------------------------------
# 7. Recommend Phones Within User Budget
# ---------------------------------------------------------

budget = 20000

recommendations = df[df["Price"] <= budget]

print("\nRealme Phones Within Budget ₹", budget)

print(recommendations.to_string(index=False))

# ---------------------------------------------------------
# 8. Predict Price for User-Entered Specifications
# ---------------------------------------------------------

print("\nEnter Your Realme Phone Specifications")

ram = int(input("RAM (GB): "))
storage = int(input("Storage (GB): "))
battery = int(input("Battery (mAh): "))
camera = int(input("Camera (MP): "))
fiveg = int(input("5G support (1=Yes, 0=No): "))

user_phone = pd.DataFrame([{
    "RAM": ram,
    "Storage": storage,
    "Battery": battery,
    "Camera": camera,
    "5G": fiveg
}])

user_price = model.predict(user_phone)[0]

print("\nEstimated Realme Smartphone Price: ₹",
      round(user_price))
