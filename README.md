# Credit Card Fraud Detection
<img width="548" height="401" alt="UI" src="https://github.com/user-attachments/assets/11503521-0744-4428-b66d-d1c37ae674e8" />


An end-to-end machine learning project for detecting potentially fraudulent credit card transactions using supervised learning, imbalance-handling techniques, Random Forest, FastAPI, and a web-based frontend.

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small proportion of all transactions.

This project builds a complete fraud detection system that:

* Performs exploratory data analysis (EDA)
* Preprocesses transaction data
* Handles severe class imbalance
* Trains and compares multiple machine learning models
* Evaluates models using fraud-focused metrics
* Uses Random Forest for prediction
* Saves the trained model
* Exposes the model through a FastAPI REST API
* Provides an interactive web frontend for predictions

The final system allows a user to provide a transaction and receive a prediction of whether it is **Fraud** or **Legitimate**, together with a fraud probability.

---

## 🏗️ System Architecture

```text
                    Credit Card Dataset
                           │
                           ▼
                    Exploratory Data
                       Analysis
                           │
                           ▼
                    Data Preprocessing
                           │
                           ▼
                 Train / Test Split
                           │
                           ▼
              ┌─────────────────────────┐
              │   Machine Learning      │
              │                         │
              │ Logistic Regression     │
              │ Decision Tree           │
              │ Random Forest           │
              └─────────────────────────┘
                           │
                           ▼
                 Imbalance Handling
                Class Weight / SMOTE
                           │
                           ▼
                  Model Evaluation
                           │
                           ▼
                   Random Forest
                           │
                           ▼
                  Saved ML Model
                    (.pkl file)
                           │
                           ▼
                    FastAPI Backend
                           │
                           ▼
                    Web Frontend
                           │
                           ▼
              Fraud / Legitimate Result
                 + Fraud Probability
```

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

The dataset contains transactions made by European cardholders and includes:

* `Time` — time elapsed between transactions
* `V1` to `V28` — anonymized numerical features
* `Amount` — transaction amount
* `Class` — target variable

Target:

```text
0 → Legitimate transaction
1 → Fraudulent transaction
```

### Dataset Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> The dataset itself is not included in this repository because of its size. Download it from Kaggle and place `creditcard.csv` inside the `data/` directory.

---

## 🔎 Exploratory Data Analysis

The project performs several EDA steps, including:

* Dataset shape and structure
* Data type inspection
* Missing-value checking
* Class distribution analysis
* Transaction amount analysis
* Statistical summaries
* Correlation analysis
* Feature relationship analysis

A major finding was the **severe class imbalance** between legitimate and fraudulent transactions.

Because of this imbalance, accuracy alone is not a reliable measure of model performance.

---

## ⚙️ Data Preprocessing

The preprocessing workflow includes:

1. Separating features and target
2. Stratified train/test splitting
3. Feature scaling where appropriate
4. Handling class imbalance
5. Preventing information leakage by applying transformations using training data

The dataset is split using stratification so that the fraud/legitimate class distribution is maintained between training and testing sets.

---

## 🤖 Machine Learning Models

Several classification algorithms were explored:

### Logistic Regression

Used as a baseline classification model.

### Decision Tree

Used to capture nonlinear relationships between transaction features.

### Random Forest

An ensemble of decision trees used as the main prediction model.

Random Forest was selected for the final application because it provides a strong baseline for tabular classification and can model nonlinear relationships.

---

## ⚖️ Handling Class Imbalance

Because fraudulent transactions are rare, several approaches were explored:

### Class Weighting

Models were trained with class weights to give greater importance to the minority fraud class.

### SMOTE

Synthetic Minority Oversampling Technique was also experimented with on the training data to increase representation of the minority class.

SMOTE was applied only to the training data to avoid contaminating the test set.

---

## 📈 Model Evaluation

The models were evaluated using metrics that are more meaningful for fraud detection:

### Precision

Of the transactions predicted as fraud, how many were actually fraudulent?

### Recall

Of all actual fraudulent transactions, how many did the model successfully detect?

### F1 Score

A balance between precision and recall.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### PR-AUC

Precision-Recall AUC is particularly useful for highly imbalanced classification problems.

---

## 🚀 FastAPI Backend

The trained model is exposed through a FastAPI REST API.

### API Endpoints

| Method | Endpoint   | Description                                  |
| ------ | ---------- | -------------------------------------------- |
| GET    | `/`        | Checks whether the API is running            |
| GET    | `/sample`  | Returns a sample transaction                 |
| POST   | `/predict` | Predicts whether a transaction is fraudulent |

Interactive API documentation is available through Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 🖥️ Frontend

The project includes a simple web interface built using:

* HTML
* CSS
* JavaScript

The frontend allows users to:

* Load a sample transaction
* View transaction information
* Submit a transaction for prediction
* See the predicted class
* View the fraud probability
* Expand advanced V1–V28 features when required

### Frontend → Backend Flow

```text
User
 │
 ▼
Web Interface
 │
 ▼
JavaScript Fetch Request
 │
 ▼
FastAPI /predict
 │
 ▼
Random Forest Model
 │
 ▼
Prediction + Probability
 │
 ▼
Frontend Result
```

---

## 📁 Project Structure

```text
Credit_Card-1st/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── app/
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── fraud_detection_model.pkl
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

> The dataset should remain local and should not be committed to GitHub.

---

## 🛠️ Technologies Used

### Programming

* Python
* HTML
* CSS
* JavaScript

### Data Science

* NumPy
* Pandas
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Imbalanced-learn
* SMOTE
* Random Forest
* Logistic Regression
* Decision Tree

### Deployment / API

* FastAPI
* Uvicorn
* Joblib

---

## ⚡ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Credit_Card-1st
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle and place:

```text
creditcard.csv
```

inside:

```text
data/
```

---

## ▶️ Running the Application

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Open the frontend:

```text
frontend/index.html
```

Then use **Load Sample Transaction** and click **Check Transaction**.

---

## 🔐 Important Notes

The `V1–V28` features in the dataset are anonymized features. They do not represent directly interpretable information such as merchant name, card number, location, or customer identity.

The fraud probability displayed by the application is the model's estimated probability for the positive class. It should not be interpreted as a guaranteed determination of fraud.

This project is intended for educational and portfolio purposes and is not a production banking fraud detection system.

---

## 🎯 Learning Outcomes

Through this project, I practiced:

* Exploratory Data Analysis
* Data preprocessing
* Imbalanced classification
* Model comparison
* Class weighting
* SMOTE
* Cross-validation
* Hyperparameter tuning
* Classification metrics
* Feature importance
* Model serialization
* REST API development
* Frontend/backend integration
* End-to-end machine learning deployment

---

## 🔮 Future Improvements

Potential improvements include:

* Threshold optimization based on business costs
* More extensive hyperparameter optimization
* PR-AUC-based model selection
* Explainable AI using SHAP
* Better transaction input abstraction
* Authentication and API security
* Cloud deployment
* Model monitoring
* Data drift detection
* Automated retraining pipelines

---

## 👨‍💻 Author

**Akshay Agarwal**

MSc Artificial Intelligence

University of East London
