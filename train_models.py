"""Train and export the car price prediction models.

This script is the training step. Run it whenever the dataset changes:
    python3 train_models.py

It reads car_data.csv, trains four linear regression models with different
training-set sizes, evaluates each model, and saves the model artifacts as JSON
files in the models/ folder.
"""

from pathlib import Path
import csv
import json
import math
import random


DATA_FILE = Path("car_data.csv")
MODELS_DIR = Path("models")
RANDOM_SEED = 42
MODEL_CONFIGS = [
    ("model_70", "Model - 70", 0.70),
    ("model_75", "Model 75", 0.75),
    ("model_80", "Model 80", 0.80),
    ("model_85", "Model 85", 0.85),
]


def load_cars():
    with DATA_FILE.open(newline="") as file:
        reader = csv.DictReader(file)
        return [
            {
                "car_id": int(row["car_id"]),
                "mileage_km": float(row["mileage_km"]),
                "age_years": float(row["age_years"]),
                "price_usd": float(row["price_usd"]),
            }
            for row in reader
        ]


def solve_linear_system(matrix, values):
    size = len(values)
    augmented = [row[:] + [values[index]] for index, row in enumerate(matrix)]

    for pivot_index in range(size):
        best_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        augmented[pivot_index], augmented[best_row] = (
            augmented[best_row],
            augmented[pivot_index],
        )

        pivot = augmented[pivot_index][pivot_index]
        if abs(pivot) < 1e-12:
            raise ValueError("The training data cannot produce a stable model.")

        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot

        for row_index in range(size):
            if row_index == pivot_index:
                continue

            factor = augmented[row_index][pivot_index]
            for column in range(pivot_index, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot_index][column]

    return [augmented[row_index][size] for row_index in range(size)]


def train_model(cars):
    """Train a linear regression model: price = b0 + b1*mileage + b2*age."""
    feature_rows = [[1.0, car["mileage_km"], car["age_years"]] for car in cars]
    prices = [car["price_usd"] for car in cars]

    xtx = [
        [
            sum(row[left] * row[right] for row in feature_rows)
            for right in range(3)
        ]
        for left in range(3)
    ]
    xty = [
        sum(row[column] * price for row, price in zip(feature_rows, prices))
        for column in range(3)
    ]

    return solve_linear_system(xtx, xty)


def split_train_test(cars, train_size, seed=RANDOM_SEED):
    shuffled = cars[:]
    random.Random(seed).shuffle(shuffled)
    training_count = round(len(shuffled) * train_size)
    return shuffled[:training_count], shuffled[training_count:]


def predict_price(coefficients, mileage_km, age_years):
    intercept, mileage_coefficient, age_coefficient = coefficients
    return intercept + (mileage_coefficient * mileage_km) + (age_coefficient * age_years)


def evaluate_model(coefficients, cars):
    actual_prices = [car["price_usd"] for car in cars]
    predicted_prices = [
        predict_price(coefficients, car["mileage_km"], car["age_years"]) for car in cars
    ]
    errors = [
        actual - predicted
        for actual, predicted in zip(actual_prices, predicted_prices)
    ]

    mean_actual = sum(actual_prices) / len(actual_prices)
    total_error = sum((actual - mean_actual) ** 2 for actual in actual_prices)
    residual_error = sum(error ** 2 for error in errors)

    return {
        "rows": len(cars),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(residual_error / len(errors)),
        "r2": 1 - (residual_error / total_error) if total_error else 0,
    }


def prediction_rows(coefficients, cars):
    return [
        {
            "car_id": car["car_id"],
            "actual_price": car["price_usd"],
            "predicted_price": predict_price(
                coefficients,
                car["mileage_km"],
                car["age_years"],
            ),
        }
        for car in cars
    ]


def scale(value, low, high, output_low, output_high):
    if high == low:
        return (output_low + output_high) / 2
    return output_low + ((value - low) / (high - low)) * (output_high - output_low)


def save_prediction_plot(model_data, predictions):
    width = 900
    height = 600
    margin_left = 82
    margin_top = 54
    margin_right = 34
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    prices = [
        price
        for row in predictions
        for price in (row["actual_price"], row["predicted_price"])
    ]
    lowest_price = min(prices)
    highest_price = max(prices)
    padding = (highest_price - lowest_price) * 0.08
    lowest_price = max(0, lowest_price - padding)
    highest_price += padding

    def point(actual, predicted):
        x = scale(actual, lowest_price, highest_price, margin_left, margin_left + plot_width)
        y = scale(predicted, lowest_price, highest_price, margin_top + plot_height, margin_top)
        return x, y

    low_x, low_y = point(lowest_price, lowest_price)
    high_x, high_y = point(highest_price, highest_price)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5"><title>Car {row["car_id"]}: actual ${row["actual_price"]:,.0f}, predicted ${row["predicted_price"]:,.0f}</title></circle>'
        for row in predictions
        for x, y in [point(row["actual_price"], row["predicted_price"])]
    )

    tick_values = [
        lowest_price + ((highest_price - lowest_price) * index / 4)
        for index in range(5)
    ]
    x_ticks = "\n".join(
        f"""
        <g>
          <line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 6}" />
          <text x="{x:.2f}" y="{margin_top + plot_height + 28}">${value:,.0f}</text>
        </g>
        """
        for value in tick_values
        for x in [scale(value, lowest_price, highest_price, margin_left, margin_left + plot_width)]
    )
    y_ticks = "\n".join(
        f"""
        <g>
          <line x1="{margin_left - 6}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" />
          <text x="{margin_left - 12}" y="{y + 5:.2f}">${value:,.0f}</text>
        </g>
        """
        for value in tick_values
        for y in [scale(value, lowest_price, highest_price, margin_top + plot_height, margin_top)]
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{model_data["label"]} actual vs predicted prices</title>
  <desc id="desc">Scatter plot comparing actual test-set prices with predicted prices.</desc>
  <rect width="100%" height="100%" fill="#ffffff" />
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #17202a; }}
    .muted {{ fill: #657383; font-size: 14px; }}
    .axis {{ stroke: #657383; stroke-width: 1.4; }}
    .grid {{ stroke: #d8e0e8; stroke-width: 1; }}
    .perfect {{ stroke: #b42318; stroke-width: 2.2; stroke-dasharray: 8 7; }}
    circle {{ fill: #0f766e; stroke: #17202a; stroke-width: 1; opacity: 0.88; }}
  </style>
  <text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-weight="700">{model_data["label"]}: Test Set Actual vs Predicted Prices</text>
  <text x="{width / 2}" y="52" text-anchor="middle" class="muted">Perfect predictions fall on the red dashed line.</text>
  <g class="grid">
    {''.join(f'<line x1="{margin_left}" y1="{scale(value, lowest_price, highest_price, margin_top + plot_height, margin_top):.2f}" x2="{margin_left + plot_width}" y2="{scale(value, lowest_price, highest_price, margin_top + plot_height, margin_top):.2f}" />' for value in tick_values)}
    {''.join(f'<line x1="{scale(value, lowest_price, highest_price, margin_left, margin_left + plot_width):.2f}" y1="{margin_top}" x2="{scale(value, lowest_price, highest_price, margin_left, margin_left + plot_width):.2f}" y2="{margin_top + plot_height}" />' for value in tick_values)}
  </g>
  <line class="perfect" x1="{low_x:.2f}" y1="{low_y:.2f}" x2="{high_x:.2f}" y2="{high_y:.2f}" />
  <g class="axis">
    <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" />
    <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" />
  </g>
  <g class="muted" text-anchor="middle">{x_ticks}</g>
  <g class="muted" text-anchor="end">{y_ticks}</g>
  <g>{circles}</g>
  <text x="{margin_left + plot_width / 2}" y="{height - 24}" text-anchor="middle" class="muted">Actual Price (USD)</text>
  <text transform="translate(24 {margin_top + plot_height / 2}) rotate(-90)" text-anchor="middle" class="muted">Predicted Price (USD)</text>
</svg>
"""
    plot_file = MODELS_DIR / model_data["plot_file"]
    plot_file.write_text(svg, encoding="utf-8")


def build_model_artifact(key, label, train_size, cars):
    training_cars, testing_cars = split_train_test(cars, train_size)
    coefficients = train_model(training_cars)
    test_size = round(1 - train_size, 2)

    return {
        "key": key,
        "label": label,
        "algorithm": "linear_regression",
        "features": ["mileage_km", "age_years"],
        "target": "price_usd",
        "train_size": round(train_size, 2),
        "test_size": test_size,
        "train_percent": round(train_size * 100),
        "test_percent": round(test_size * 100),
        "random_seed": RANDOM_SEED,
        "plot_file": f"{key}_plot.svg",
        "coefficients": {
            "intercept": coefficients[0],
            "mileage_km": coefficients[1],
            "age_years": coefficients[2],
        },
        "training_metrics": evaluate_model(coefficients, training_cars),
        "testing_metrics": evaluate_model(coefficients, testing_cars),
    }


def main():
    cars = load_cars()
    MODELS_DIR.mkdir(exist_ok=True)

    for key, label, train_size in MODEL_CONFIGS:
        artifact = build_model_artifact(key, label, train_size, cars)
        coefficients = [
            artifact["coefficients"]["intercept"],
            artifact["coefficients"]["mileage_km"],
            artifact["coefficients"]["age_years"],
        ]
        _training_cars, testing_cars = split_train_test(cars, train_size)
        save_prediction_plot(artifact, prediction_rows(coefficients, testing_cars))
        output_file = MODELS_DIR / f"{key}.json"
        output_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
