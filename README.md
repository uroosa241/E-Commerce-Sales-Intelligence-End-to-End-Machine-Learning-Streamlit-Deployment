#  E-Commerce Sales Intelligence

### End-to-End Machine Learning Regression & Streamlit Deployment

An end-to-end **Machine Learning project for e-commerce sales prediction**, built with Python, Scikit-learn, Pandas, and Streamlit.

The project demonstrates the complete machine learning workflow — from data preprocessing and feature engineering to model training, evaluation, prediction, and deployment through an interactive web application.

---

##  Project Overview

The objective of this project is to predict **e-commerce order sales** based on product, order source, location, quantity, and date-related features.

The application provides a professional interactive dashboard where users can:

* Explore overall sales performance
* Analyze sales by category and order source
* View top-performing products
* Enter order information and generate a sales prediction
* Explore and filter the dataset
* Download filtered data
* Review model performance
* Inspect the machine learning feature architecture

The deployed application uses a **Random Forest Regressor** integrated into a complete Scikit-learn preprocessing and modeling pipeline.

---

##  Machine Learning Objective

**Problem Type:** Regression

**Target Variable:** `sales`

### Input Features

#### Numerical Features

* `quantity`
* `year`
* `month`
* `day`
* `day_of_week`

#### Categorical Features

* `order_source`
* `category`
* `sku`
* `city`

The project intentionally excludes `order_id` and `order_status` from the prediction features to avoid unnecessary information and potential leakage.

---

##  Machine Learning Pipeline

The project follows a production-oriented preprocessing pipeline:

```text
Raw E-Commerce Data
        ↓
Data Cleaning
        ↓
Date Conversion
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Numerical Preprocessing
        ↓
Categorical Preprocessing
        ↓
Random Forest Regressor
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
Streamlit Application
```

### Numerical preprocessing

```text
Missing Values
      ↓
Median Imputation
      ↓
StandardScaler
```

### Categorical preprocessing

```text
Missing Values
      ↓
Most-Frequent Imputation
      ↓
OneHotEncoder
```

The preprocessing and model are combined into a single Scikit-learn `Pipeline`, making the trained model easier to deploy consistently.

---

##  Model

### Random Forest Regressor

The project uses:

```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)
```

The model is trained using an **80/20 train-test split**.

---

##  Model Evaluation

The application evaluates the regression model using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted sales.

### RMSE — Root Mean Squared Error

Measures prediction error while giving larger errors more weight.

### R² Score

Measures how well the model explains the variation in the target variable.

The Streamlit application displays these metrics dynamically under the **Model Performance** section.

---

##  Streamlit Application

The application contains four main sections:

### 1.  Executive Dashboard

Provides an overview of:

* Total Sales
* Average Order Value
* Median Order Value
* Total Orders
* Units Sold
* Sales by Product Category
* Sales by Order Source
* Top Products

### 2.  Sales Prediction

Users can enter:

* Order Source
* Category
* SKU / Product
* Quantity
* City
* Order Date

The application automatically derives:

* Year
* Month
* Day
* Day of Week

The trained model then generates the estimated sales value.

### 3.  Data Explorer

Users can:

* View the dataset
* Filter by city
* Filter by category
* Inspect records
* Download filtered data as CSV

### 4. Model Performance

Displays:

* MAE
* RMSE
* R²
* Training rows
* Test rows
* Feature architecture
* Deployment model artifact status

---

##  User Interface

The application uses a clean, professional light dashboard design:

* White background
* Navy typography
* Blue primary actions
* Teal/green status indicators
* Responsive Streamlit layout
* KPI cards
* Interactive charts
* Prediction result cards
* Professional sidebar navigation

The design is intended to resemble a modern business intelligence and machine learning application.

---

##  Tech Stack

| Technology                    | Purpose                     |
| ----------------------------- | --------------------------- |
| Python                        | Core programming language   |
| Pandas                        | Data manipulation           |
| NumPy                         | Numerical computation       |
| Scikit-learn                  | Machine learning            |
| Streamlit                     | Interactive web application |
| Joblib                        | Model serialization         |
| Matplotlib / Streamlit Charts | Data visualization          |

---

##  Project Structure

```text
Ecommerce-ML-Project/
│
├── app.py
├── eCommercePK.csv
├── ecommerce_sales_model.joblib
├── requirements.txt
├── README.md
│
└── notebooks/
    └── ecommerce_sales_ml.ipynb
```

> File names may vary depending on the final project organization.

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ecommerce-sales-intelligence.git
```

Navigate into the project:

```bash
cd ecommerce-sales-intelligence
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

##  Requirements

The main dependencies are:

```text
pandas
numpy
scikit-learn
streamlit
joblib
```

---

##  Run the Application

From the project directory:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Model Deployment

The trained machine learning pipeline is serialized using Joblib:

```python
joblib.dump(pipeline, MODEL_PATH)
```

The application therefore maintains the complete preprocessing + model pipeline rather than saving only the estimator.

This helps ensure that the same preprocessing steps used during training are applied during prediction.

---

## Future Improvements

Potential extensions for this project include:

* Automated Excel/CSV upload
* Automatic feature detection
* Automated classification/regression detection
* Hyperparameter optimization
* Model comparison dashboard
* Cross-validation
* Feature importance visualization
* SHAP explainability
* Prediction confidence / uncertainty analysis
* Automated data quality checks
* Docker containerization
* FastAPI prediction API
* Cloud deployment
* Database integration
* Automated ML pipeline
* Model monitoring

---

##  Deployment

The application is designed for deployment using Streamlit-compatible hosting.

A future production architecture could be:

```text
User
  ↓
Streamlit Interface
  ↓
FastAPI / ML API
  ↓
Preprocessing Pipeline
  ↓
Trained ML Model
  ↓
Prediction
```

For larger production datasets, the architecture can be extended to use databases or analytical engines instead of loading the complete dataset directly into memory.

---

##  Key Learning Outcomes

This project demonstrates practical experience with:

* Data preprocessing
* Feature engineering
* Numerical feature scaling
* Categorical encoding
* Missing-value handling
* Train-test splitting
* Regression modeling
* Random Forest
* Model evaluation
* Scikit-learn Pipelines
* Model serialization
* Streamlit application development
* Interactive dashboards
* ML model deployment concepts

---

##  Author

**Uroosa Khan**

Data & Business Intelligence | Machine Learning | Python | SQL | Power BI

GitHub: `https://github.com/uroosa241`

---

##  Project Goal

This project is part of my journey toward building **production-oriented Machine Learning applications**, combining machine learning modeling with data analytics, interactive interfaces, and deployment.

If you find this project useful, feel free to ⭐ the repository.
