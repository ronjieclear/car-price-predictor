# Car Price Predictor

A Python web app that predicts estimated used-car prices from mileage and age. The app loads exported linear regression model files, lets the user choose a train/test split model, and displays the predicted price together with model evaluation metrics and a plot.

## Online Demo

Live app: [https://ml.averxon.com](https://ml.averxon.com)

## Features

- Predicts used-car price from:
  - Mileage in kilometers
  - Age in years
- Supports multiple exported model versions:
  - 70/30 train-test split
  - 75/25 train-test split
  - 80/20 train-test split
  - 85/15 train-test split
- Shows model details, including:
  - Training and testing row counts
  - Mean absolute error
  - Root mean squared error
  - R-squared score
- Serves model plot SVG files directly from the app.
- Runs without a database or external API.

## Tech Stack

- Python 3
- Built-in `http.server`
- CSV data source
- Exported JSON model files
- Linear regression model coefficients

The training scripts use:

- pandas
- scikit-learn

## Project Structure

```text
.
├── app.py
├── car_data.csv
├── requirements.txt
├── models/
│   ├── model_70.json
│   ├── model_70_plot.svg
│   ├── model_75.json
│   ├── model_75_plot.svg
│   ├── model_80.json
│   ├── model_80_plot.svg
│   ├── model_85.json
│   └── model_85_plot.svg
├── train_models.py
├── LICENSE
└── README.md
```

## How It Works

The web app reads `car_data.csv` and model files from the `models/` folder. Each model JSON file contains:

- Selected features
- Target column
- Train/test split
- Linear regression coefficients
- Evaluation metrics
- Linked SVG plot file

When a user submits mileage and age, the app calculates:

```text
predicted_price = intercept + mileage_coefficient * mileage_km + age_coefficient * age_years
```

The prediction is displayed as a dollar estimate.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/ronjieclear/car-price-predictor.git
cd car-price-predictor
```

Optional: create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install training dependencies:

```bash
pip install -r requirements.txt
```

Start the web app:

```bash
python3 app.py
```

Open:

```text
http://localhost:8000
```

## Deployment

This app is ready for deployment on DigitalOcean App Platform.

Recommended settings:

```text
Resource type: Web Service
Run command: python app.py
HTTP port: 8000
```

The app reads deployment environment variables:

```python
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
```

This allows it to run locally and on a hosted server.

## Data And Model Notes

This project is intended as a machine learning demonstration. Predictions are based on the included dataset and exported linear regression coefficients. The result should be treated as an estimate, not as a professional vehicle appraisal.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
