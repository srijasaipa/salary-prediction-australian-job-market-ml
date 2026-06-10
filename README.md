<div align="center">

# 🇦🇺 Australian Job Salary Prediction

### Machine Learning-powered salary prediction for the Australian job market

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Project Structure](#-project-structure)

</div>

---

## 📋 Overview

A comprehensive machine learning solution for predicting median salaries in the Australian job market. This project combines exploratory data analysis, a robust Random Forest regression pipeline, and an interactive Streamlit web application to provide accurate salary predictions based on job characteristics, location, and market conditions.

## ✨ Features

- 🔍 **Comprehensive EDA** - Feature correlation analysis and data leakage detection
- 🤖 **ML Pipeline** - Production-ready scikit-learn pipeline with preprocessing
- 📊 **High Accuracy** - R² score of 0.87 with only 6.61% MAPE
- 🌐 **Interactive UI** - User-friendly Streamlit web interface
- 💾 **Model Persistence** - Trained model saved with joblib for deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd salary-prediction-aus
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn streamlit joblib
```

### Usage

#### 1️⃣ Run Exploratory Data Analysis
```bash
python eda.py
```
Analyzes feature correlations and identifies potential data leakage.

#### 2️⃣ Train the Model
```bash
python train_model.py
```
Trains the Random Forest model and saves artifacts (`model_pipeline.joblib`, `model_features.joblib`).

#### 3️⃣ Launch the Web App
```bash
streamlit run app.py
```
Opens the interactive prediction interface at `http://localhost:8501`.

## 📊 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | `0.8705` | Model explains 87% of salary variance |
| **MAE** | `$5,176.66` | Mean Absolute Error |
| **RMSE** | `$6,585.95` | Root Mean Squared Error |
| **MAPE** | `6.61%` | Mean Absolute Percentage Error |

## 📁 Project Structure

```
📦 salary-prediction-aus
├── 📊 Australian_job_data_cleaned.csv    # Dataset
├── 🔍 eda.py                             # Exploratory Data Analysis
├── 🧠 train_model.py                     # Model Training Script
├── 🌐 app.py                             # Streamlit Application
├── 💾 model_pipeline.joblib              # Trained Model
├── 📋 model_features.joblib              # Feature List
└── 📖 README.md                          # Documentation
```

## 🛠️ Technical Details

### Data Preprocessing
- **Leakage Removal**: Excluded `salary_min_aud`, `salary_max_aud`, `salary_range_aud`, `salary_log_median`
- **Feature Engineering**: Handled one-hot encoding for categorical variables
- **Redundancy Elimination**: Removed `month_name`, `quarter`, `remote_availability`

### Model Architecture
- **Algorithm**: Random Forest Regressor (100 estimators)
- **Pipeline Components**:
  - `SimpleImputer` (median strategy)
  - `StandardScaler` (feature normalization)
  - `RandomForestRegressor` (ensemble learning)

### Key Features Used
- **Categorical**: Job title, industry, city, region, education, job type, skills
- **Numerical**: Experience, openings, demand index, competitiveness score
- **Binary**: Remote flag, metro city, high demand indicator

## 🎯 Use Cases

- 💼 **Job Seekers**: Estimate fair salary expectations
- 🏢 **Recruiters**: Benchmark competitive compensation
- 📈 **Market Analysis**: Understand salary trends across industries

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

⚠️ Note: The file `model_pipeline.joblib` is not included in this repository because it exceeds GitHub's web upload size limit. To generate the model locally, run: python train_model.py


**Made with ❤️ for the Australian job market**

</div>
