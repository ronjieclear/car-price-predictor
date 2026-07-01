"""Web app for predicting used-car prices from exported model files.

Professional workflow:
    1. Train/export models: python3 train_models.py
    2. Start web app:      python3 app.py
    3. Open browser:       http://localhost:8000
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import csv
import html
import json
import os


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
DATA_FILE = Path("car_data.csv")
MODELS_DIR = Path("models")
MODEL_FILES = [
    "model_70.json",
    "model_75.json",
    "model_80.json",
    "model_85.json",
]
DEFAULT_MODEL_KEY = "model_70"


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


def load_models():
    models = {}
    missing_files = []

    for filename in MODEL_FILES:
        model_file = MODELS_DIR / filename
        if not model_file.exists():
            missing_files.append(str(model_file))
            continue

        model_data = json.loads(model_file.read_text(encoding="utf-8"))
        models[model_data["key"]] = model_data

    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing model files: {missing}. Run python3 train_models.py first."
        )

    return models


def predict_price(model_data, mileage_km, age_years):
    coefficients = model_data["coefficients"]
    return (
        coefficients["intercept"]
        + (coefficients["mileage_km"] * mileage_km)
        + (coefficients["age_years"] * age_years)
    )


CARS = load_cars()
MODELS = load_models()


def money(value):
    return f"${value:,.0f}"


def number(value):
    return f"{value:,.0f}"


def get_selected_model(selected_model_key):
    return MODELS.get(selected_model_key, MODELS[DEFAULT_MODEL_KEY])


def model_options(selected_model_key):
    return "\n".join(
        f'<option value="{html.escape(key)}" {selected}>{html.escape(model["label"])}</option>'
        for key, model in MODELS.items()
        for selected in ["selected" if key == selected_model_key else ""]
    )


def metric_rows(selected_model):
    rows = [
        ("Training set", selected_model["training_metrics"]),
        ("Test set", selected_model["testing_metrics"]),
    ]
    return "\n".join(
        f"""
        <tr>
            <td>{label}</td>
            <td>{metrics["rows"]}</td>
            <td>{money(metrics["mae"])}</td>
            <td>{money(metrics["rmse"])}</td>
            <td>{metrics["r2"]:.3f}</td>
        </tr>
        """
        for label, metrics in rows
    )


def render_page(result=None, mileage="", age="", selected_model_key=DEFAULT_MODEL_KEY, error=""):
    selected_model = get_selected_model(selected_model_key)
    selected_model_key = selected_model["key"]
    coefficients = selected_model["coefficients"]
    training_rows = selected_model["training_metrics"]["rows"]
    testing_rows = selected_model["testing_metrics"]["rows"]
    latest_cars = sorted(CARS, key=lambda car: car["car_id"], reverse=True)[:5]
    plot_url = f"/model_plot.svg?model={html.escape(selected_model_key)}"

    result_html = ""
    if result is not None:
        result_html = f"""
        <section class="result-panel" aria-live="polite">
            <span class="eyebrow">Estimated price</span>
            <strong>{money(result)}</strong>
            <p>{html.escape(selected_model["label"])} prediction based on {number(float(mileage))} km mileage and {float(age):g} years of age.</p>
        </section>
        """

    error_html = ""
    if error:
        error_html = f'<p class="error">{html.escape(error)}</p>'

    rows_html = "\n".join(
        f"""
        <tr>
            <td>{car["car_id"]}</td>
            <td>{number(car["mileage_km"])}</td>
            <td>{car["age_years"]:g}</td>
            <td>{money(car["price_usd"])}</td>
        </tr>
        """
        for car in latest_cars
    )

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Car Price Predictor</title>
    <style>
        :root {{
            color-scheme: light;
            --ink: #17202a;
            --muted: #657383;
            --line: #d8e0e8;
            --panel: #ffffff;
            --bg: #f4f7f9;
            --accent: #0f766e;
            --accent-strong: #115e59;
            --warn: #b42318;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            color: var(--ink);
            background: var(--bg);
        }}

        header {{
            background: #17202a;
            color: #ffffff;
            padding: 28px 24px;
        }}

        .header-inner,
        main {{
            width: min(1120px, calc(100% - 32px));
            margin: 0 auto;
        }}

        h1 {{
            margin: 0 0 8px;
            font-size: 32px;
            letter-spacing: 0;
        }}

        header p {{
            max-width: 720px;
            margin: 0;
            color: #dce7ee;
            line-height: 1.5;
        }}

        main {{
            display: grid;
            grid-template-columns: minmax(280px, 420px) 1fr;
            gap: 20px;
            padding: 24px 0 40px;
        }}

        section {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 20px;
        }}

        h2 {{
            margin: 0 0 16px;
            font-size: 20px;
            letter-spacing: 0;
        }}

        form {{
            display: grid;
            gap: 16px;
        }}

        label {{
            display: grid;
            gap: 7px;
            color: var(--muted);
            font-weight: 700;
        }}

        input,
        select {{
            width: 100%;
            min-height: 44px;
            border: 1px solid #bac7d3;
            border-radius: 6px;
            padding: 10px 12px;
            color: var(--ink);
            background: #ffffff;
            font-size: 16px;
        }}

        input:focus,
        select:focus {{
            outline: 3px solid rgba(15, 118, 110, 0.2);
            border-color: var(--accent);
        }}

        button {{
            min-height: 46px;
            border: 0;
            border-radius: 6px;
            padding: 11px 16px;
            color: #ffffff;
            background: var(--accent);
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
        }}

        button:hover {{
            background: var(--accent-strong);
        }}

        .result-panel {{
            margin-top: 18px;
            border-color: #9ccbc5;
            background: #edf8f6;
        }}

        .result-panel strong {{
            display: block;
            margin-top: 5px;
            font-size: 36px;
            letter-spacing: 0;
        }}

        .result-panel p {{
            margin: 8px 0 0;
            color: var(--muted);
            line-height: 1.45;
        }}

        .eyebrow {{
            color: var(--accent-strong);
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .error {{
            margin: 0;
            color: var(--warn);
            font-weight: 700;
        }}

        .model-details {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 18px;
        }}

        .metric {{
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 12px;
        }}

        .metric span {{
            display: block;
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
        }}

        .metric strong {{
            display: block;
            margin-top: 4px;
            font-size: 17px;
        }}

        .plot {{
            width: 100%;
            height: auto;
            border: 1px solid var(--line);
            border-radius: 8px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 18px;
            font-size: 14px;
        }}

        th,
        td {{
            border-bottom: 1px solid var(--line);
            padding: 10px 8px;
            text-align: right;
        }}

        th:first-child,
        td:first-child,
        .evaluation-table td:first-child,
        .evaluation-table th:first-child {{
            text-align: left;
        }}

        th {{
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
        }}

        .note {{
            margin: 12px 0 0;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.45;
        }}

        @media (max-width: 820px) {{
            main {{
                grid-template-columns: 1fr;
            }}

            .model-details {{
                grid-template-columns: 1fr;
            }}

            h1 {{
                font-size: 27px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-inner">
            <h1>Car Price Predictor</h1>
            <p>Select an exported model, enter the car details, and estimate the used-car price from saved linear regression coefficients.</p>
        </div>
    </header>

    <main>
        <div>
            <section>
                <h2>Predict a Price</h2>
                <form method="post" action="/">
                    {error_html}
                    <label>
                        Model
                        <select name="model" required>
                            {model_options(selected_model_key)}
                        </select>
                    </label>
                    <label>
                        Mileage in kilometers
                        <input name="mileage" type="number" min="0" step="100" value="{html.escape(str(mileage))}" required>
                    </label>
                    <label>
                        Age in years
                        <input name="age" type="number" min="0" step="0.5" value="{html.escape(str(age))}" required>
                    </label>
                    <button type="submit">Predict Price</button>
                </form>
                {result_html}
            </section>

            <section style="margin-top: 20px;">
                <h2>Loaded Model</h2>
                <div class="model-details">
                    <div class="metric">
                        <span>Train/test split</span>
                        <strong>{training_rows}/{testing_rows}</strong>
                    </div>
                    <div class="metric">
                        <span>Mileage effect</span>
                        <strong>{coefficients["mileage_km"]:.4f}</strong>
                    </div>
                    <div class="metric">
                        <span>Age effect</span>
                        <strong>{coefficients["age_years"]:.0f}</strong>
                    </div>
                </div>
                <p class="note">These values come from the selected JSON file in the models folder.</p>
            </section>

            <section style="margin-top: 20px;">
                <h2>Model Performance</h2>
                <table class="evaluation-table">
                    <thead>
                        <tr>
                            <th>Data</th>
                            <th>Rows</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>R-squared</th>
                        </tr>
                    </thead>
                    <tbody>
                        {metric_rows(selected_model)}
                    </tbody>
                </table>
                <p class="note">MAE and RMSE are average price errors in dollars. Lower is better. R-squared is closer to 1 when the model explains more of the price variation.</p>
            </section>
        </div>

        <section>
            <h2>Selected Model Plot</h2>
            <img class="plot" src="{plot_url}" alt="Actual prices versus predicted prices for the selected model">
            <table>
                <thead>
                    <tr>
                        <th>Car ID</th>
                        <th>Mileage</th>
                        <th>Age</th>
                        <th>Price</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>"""


class CarPriceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/model_plot.svg":
            query = parse_qs(parsed_url.query)
            model_key = query.get("model", [DEFAULT_MODEL_KEY])[0]
            self.send_model_plot(model_key)
            return

        if parsed_url.path != "/":
            self.send_error(404)
            return

        self.send_html(render_page())

    def do_POST(self):
        if self.path != "/":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        form = parse_qs(body)

        selected_model_key = form.get("model", [DEFAULT_MODEL_KEY])[0]
        selected_model = get_selected_model(selected_model_key)
        mileage = form.get("mileage", [""])[0]
        age = form.get("age", [""])[0]

        try:
            mileage_value = float(mileage)
            age_value = float(age)
            if mileage_value < 0 or age_value < 0:
                raise ValueError
            predicted_price = max(
                0,
                predict_price(selected_model, mileage_value, age_value),
            )
            page = render_page(predicted_price, mileage, age, selected_model_key)
        except ValueError:
            page = render_page(
                mileage=mileage,
                age=age,
                selected_model_key=selected_model_key,
                error="Please enter non-negative numbers for mileage and age.",
            )

        self.send_html(page)

    def send_html(self, page):
        encoded = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_model_plot(self, model_key):
        selected_model = get_selected_model(model_key)
        plot_file = MODELS_DIR / selected_model["plot_file"]

        if not plot_file.exists():
            self.send_error(404)
            return

        image = plot_file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
        self.send_header("Content-Length", str(len(image)))
        self.end_headers()
        self.wfile.write(image)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), CarPriceHandler)
    print(f"Car price web app running at http://{HOST}:{PORT}")
    server.serve_forever()
