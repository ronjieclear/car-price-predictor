"""Beginner example: predict used-car prices with linear regression."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Read the car dataset.
cars = pd.read_csv("car_data.csv")

# 2. Separate the input data from the answer we want to predict.
# X contains the clues given to the model: mileage and age.
# y contains the correct answer for each car: its price.
# The car_id is not included because it is only a label, not a price clue.
X = cars[["mileage_km", "age_years"]]
y = cars["price_usd"]

# 3. Split the rows into 80% training data and 20% testing data.
# Training data teaches the model the relationship between mileage, age, and price.
# Testing data is kept separate so we can check the model using unseen cars.
# random_state=42 keeps the same cars in each group every time the code runs.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,# 20% po yan  1,302 50%- 1916 25%-1,235 30%-1,264
    random_state=3,  #SEED
)

# 4. Create and train the linear regression model.
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Ask the model to predict prices for the test data.
predicted_prices = model.predict(X_test)

# 6. Show information about the results.
print("Total rows:", len(cars))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print()

# Show which original cars were placed in each group.
training_cars = cars.loc[X_train.index]
testing_cars = cars.loc[X_test.index]

print("Car IDs used for training:")
print(training_cars["car_id"].tolist())
print("\nFirst 5 training rows:")
print(training_cars.head().to_string(index=False))

print("\nCar IDs used for testing:")
print(testing_cars["car_id"].tolist())
print("\nFirst 5 testing rows:")
print(testing_cars.head().to_string(index=False))
print()

print("Intercept:", round(model.intercept_, 2))
print("Mileage coefficient:", round(model.coef_[0], 4))
print("Age coefficient:", round(model.coef_[1], 2))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predicted_prices), 2))
print("R-squared:", round(r2_score(y_test, predicted_prices), 3))

# 7. Compare the actual and predicted prices in a small table.
results = pd.DataFrame(
    {
        "Car ID": testing_cars["car_id"],
        "Actual Price": y_test,
        "Predicted Price": predicted_prices,
        "Difference": y_test - predicted_prices,
    }
)

# Remove decimal places from prices when displaying the table.
results["Actual Price"] = results["Actual Price"].astype(int)
results["Predicted Price"] = results["Predicted Price"].round().astype(int)
results["Difference"] = results["Difference"].round().astype(int)

print("\nTest results:")
print(results.to_string(index=False))
print("\nDifference = Actual Price - Predicted Price")
print("Positive difference: the model predicted too low")
print("Negative difference: the model predicted too high")

# 8. Draw a graph of actual prices versus predicted prices.
plt.scatter(y_test, predicted_prices, color="dodgerblue", edgecolor="black")

# This dashed line represents perfect predictions.
lowest_price = min(y_test.min(), predicted_prices.min())
highest_price = max(y_test.max(), predicted_prices.max())
plt.plot(
    [lowest_price, highest_price],
    [lowest_price, highest_price],
    color="red",
    linestyle="--",
    label="Perfect prediction",
)

plt.title("Actual Car Prices vs. Predicted Car Prices")
plt.xlabel("Actual Price (USD)")
plt.ylabel("Predicted Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("car_price_plot.png", dpi=150)
plt.show()
