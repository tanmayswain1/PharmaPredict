# PharmaPredict
### AI-Driven OTC Medicine Demand Forecasting

A command-line machine learning tool that predicts which Over-the-Counter medicine will be in highest demand for a given country and date, helping retail pharmacies reduce stockouts and overstock waste.

---

## Overview

Pharmacies suffer losses from two recurring problems: running out of high-demand medicine during peak periods, and overstocking slow-moving products that expire before sale. PharmaPredict addresses this by training a classification model on historical OTC sales data and exposing it through a terminal-based interface.

Two models were built and compared:

| Model | Type | Accuracy |
|-------|------|----------|
| Random Forest (m1) | Parallel Ensemble — Baseline | ~19% |
| Gradient Boosting (m2) | Sequential Ensemble — Final | ~21% |

Gradient Boosting was selected as the final model because it corrects residual errors at each training step, making it better suited to the interaction patterns in this dataset.

---

## Requirements

- Python 3.8 or higher
- pip

Install all dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`

---

## Setup

```bash
git clone https://github.com/tanmayswain/pharmapredict.git
cd pharmapredict
pip install -r requirements.txt
```

---

## Usage

### Train the models

Run the Random Forest baseline:

```bash
python scripts/m1_random_forest_classifier.py
```

Run the Gradient Boosting model (also exports `.pkl` files):

```bash
python scripts/m2_gradient_boosting_classifier.py
```

Both scripts will print Accuracy and MAE to the terminal and display a Confusion Matrix plot.

### Run a prediction

```bash
python scripts/m2_gradient_boosting_classifier.py --country India --date 2026-04-07
```

Output:

```
Predicted high-demand product: Paracetamol
```

---

## Dataset

Located at `data/pharmacy_otc_sales_data.csv`. Contains historical OTC medicine transactions with four columns:

| Column | Description |
|--------|-------------|
| `Date` | Transaction date |
| `Product` | OTC medicine name |
| `Amount` | Units sold |
| `Country` | Region of sale |

---

## Pipeline

```
pharmacy_otc_sales_data.csv
        |
        v
Preprocessing
  - Extract Month, DayOfWeek from Date
  - LabelEncode Country and Product
        |
        v
Model Training
  - m1: Random Forest     (~19% accuracy)
  - m2: Gradient Boosting (~21% accuracy)
        |
        v
Evaluation
  - Accuracy Score, MAE, Confusion Matrix
        |
        v
Export: medicine_model.pkl, country_encoder.pkl, product_encoder.pkl
        |
        v
CLI Inference
```

---

## Project Structure

```
PharmaPredict/
├── data/
│   └── pharmacy_otc_sales_data.csv
├── models/
│   ├── medicine_model.pkl
│   ├── country_encoder.pkl
│   └── product_encoder.pkl
├── scripts/
│   ├── m1_random_forest_classifier.py
│   └── m2_gradient_boosting_classifier.py
├── requirements.txt
└── README.md
```

---

## Course

CSA2001 — Fundamentals in AI and ML

**Author:** Tanmay Swain | Reg No: 25BAI11062 | B.Tech CSE | March 2026
