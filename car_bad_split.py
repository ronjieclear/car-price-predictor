"""Example of a poor train/test split that can produce a poor R-squared score."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Read the same car dataset.
cars = pd.read_csv("car_data.csv")

# 2. Sort the cars from youngest to oldest.
# This is intentionally done to create an unfair, biased split.
cars = cars.sort_values(by=["age_years", "mileage_km"])

# 3. Create a POOR split.
# The model trains only on the 50 youngest cars.
# It is then tested on the 50 oldest cars, which are different from its training data.
# A good split normally mixes the rows randomly instead of separating them by age.
training_cars = cars.iloc[:50]
testing_cars = cars.iloc[50:]

# 4. Separate the input columns from the prices.
X_train = training_cars[["mileage_km", "age_years"]]
y_train = training_cars["price_usd"]

X_test = testing_cars[["mileage_km", "age_years"]]
y_test = testing_cars["price_usd"]

# 5. Train the model using only the younger cars.
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict the prices of the older cars.
predicted_prices = model.predict(X_test)

# 7. Display details about the biased split.
print("POOR SPLIT: youngest cars for training, oldest cars for testing")
print("Training rows:", len(training_cars))
print("Training ages:", training_cars["age_years"].min(), "to", training_cars["age_years"].max())
print("Testing rows:", len(testing_cars))
print("Testing ages:", testing_cars["age_years"].min(), "to", testing_cars["age_years"].max())

print("\nCar IDs used for training:")
print(training_cars["car_id"].tolist())

print("\nCar IDs used for testing:")
print(testing_cars["car_id"].tolist())

# 8. Calculate the model's test results.
mae = mean_absolute_error(y_test, predicted_prices)
r_squared = r2_score(y_test, predicted_prices)

print("\nMean Absolute Error:", round(mae, 2))
print("R-squared:", round(r_squared, 3))

# 9. Show actual and predicted prices without decimal places.
results = pd.DataFrame(
    {
        "Car ID": testing_cars["car_id"],
        "Age": testing_cars["age_years"],
        "Actual Price": y_test,
        "Predicted Price": predicted_prices,
    }
)

results["Actual Price"] = results["Actual Price"].astype(int)
results["Predicted Price"] = results["Predicted Price"].round().astype(int)

print("\nTest results:")
print(results.to_string(index=False))

# 10. Plot actual prices against predicted prices.
plt.scatter(y_test, predicted_prices, color="orange", edgecolor="black")

# The red line shows where perfect predictions would appear.
lowest_price = min(y_test.min(), predicted_prices.min())
highest_price = max(y_test.max(), predicted_prices.max())
plt.plot(
    [lowest_price, highest_price],
    [lowest_price, highest_price],
    color="red",
    linestyle="--",
    label="Perfect prediction",
)

plt.title("Poor Split: Actual vs. Predicted Car Prices")
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("car_bad_split_plot.png", dpi=150)
plt.show()
