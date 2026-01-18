# India House Price Prediction

ML regression system for predicting residential property prices across India using historical market data.

---

## Problem Statement

Many industries rely on accurate numerical predictions to support decision-making. Whether forecasting sales, predicting customer spending, estimating delivery times, or determining equipment failure patterns, regression models play a central role. This project focuses on **Option B: House Price Prediction** - estimating property prices using features like location, size, rooms, amenities, and historical trends.

---

## Objective

Build an end-to-end ML pipeline that:

- Ingests and preprocesses structured housing data
- Identifies correlations and relevant features
- Trains, optimizes, and evaluates regression models
- Exposes the model through an API or simple UI
- Generates insights and visual explanations about predictions

---

## Dataset

**Source:** Kaggle - `ankushpanday1/india-house-price-prediction`

The dataset contains residential property records from various Indian cities with features including area, location, number of bedrooms/bathrooms, parking, furnishing status, and transaction prices.

**Dataset Size:**
- Total Samples: 250,000
- Training Set: 200,000 samples
- Test Set: 50,000 samples
- Original Features: 23
- Engineered Features: 21 (after preprocessing)

---

## Tech Stack

**Core Libraries:**
- Python 3.9+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

**ML Models:**
- Linear Regression
- Random Forest Regressor
- XGBoost

**Deployment:**
- Flask for REST API (Port 8000)
- Streamlit for web UI (Port 8501)

---

## Pipeline Architecture

```
Data Ingestion → Preprocessing → Feature Engineering → Model Training 
→ Evaluation → Deployment (API/UI)
```

**1. Data Ingestion**
- Load CSV from `data/raw/india_housing_prices.csv`
- 250,000 property records with 23 features

**2. Preprocessing**
- Handle missing values (None found in dataset)
- Detect and remove outliers using IQR method
- Encode categorical variables
- Drop high-cardinality features (Locality, Amenities)

**3. Feature Engineering**
- Correlation analysis
- Create derived features (price per sq ft, property age)
- Feature selection
- Final feature set: 21 predictive features

**4. Model Training**
- Train three regression algorithms (Linear Regression, Random Forest, XGBoost)
- Evaluate on hold-out test set
- Select best model based on R² score

**5. Deployment**
- Serialize best model (model.pkl)
- Flask REST API on port 8000
- Interactive Streamlit UI on port 8501

---

## Installation

```bash
git clone <repo-url>
cd india-house-price-prediction

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Step 1: Preprocess Data
```bash
python src/preprocess.py
```
**Output:**
- Cleaned dataset: `data/processed/housing_cleaned.csv`
- Train/test split saved to `data/processed/`

### Step 2: Train Models
```bash
python src/train.py
```
**Output:**
- Best model: `models/model.pkl`
- Metrics comparison: `models/model_comparison.csv`

### Step 3: Start API Server
```bash
python api/app.py
```
**API runs at:** `http://localhost:8000`

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions

### Step 4: Run Web Interface
```bash
streamlit run ui/streamlit_app.py
```
**UI opens at:** `http://localhost:8501`

---

## Making Predictions

### API Request Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "State": "Maharashtra",
    "City": "Mumbai",
    "Property_Type": "Apartment",
    "BHK": 3,
    "Size_in_SqFt": 1500,
    "Year_Built": 2015,
    "Furnished_Status": "Furnished",
    "Floor_No": 5,
    "Total_Floors": 10,
    "Nearby_Schools": 3,
    "Nearby_Hospitals": 2,
    "Public_Transport_Accessibility": "High",
    "Parking_Space": "Yes",
    "Security": "Yes",
    "Facing": "East",
    "Owner_Type": "First",
    "Availability_Status": "Ready to Move"
  }'
```

### API Response Example
```json
{
  "predicted_price_lakhs": 285.34,
  "predicted_price_inr": "₹2,85,34,000",
  "model_used": "Random Forest",
  "confidence": "High (R² = 0.996)"
}
```

---

## Project Structure

```
india-house-price-prediction/
│
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv    # Original dataset
│   └── processed/
│       ├── housing_cleaned.csv         # Preprocessed data
│       ├── train.csv                   # Training set
│       └── test.csv                    # Test set
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Exploratory analysis
│   └── 02_Training.ipynb              # Model experiments
│
├── src/
│   ├── preprocess.py                  # Data cleaning pipeline
│   ├── train.py                       # Model training pipeline
│   └── utils.py                       # Helper functions
│
├── models/
│   ├── model.pkl                      # Trained Random Forest model
│   └── model_comparison.csv           # Performance metrics
│
├── api/
│   └── app.py                         # Flask REST API (Port 8000)
│
├── ui/
│   └── streamlit_app.py               # Streamlit web app (Port 8501)
│
├── requirements.txt
└── README.md
```

---

## Model Performance

Three regression models were trained and evaluated on a 50,000-sample hold-out test set:

| Model | Test R² | Test RMSE (Lakhs) | Test MAE (Lakhs) | Training Time (s) | Overfitting Gap |
|-------|---------|-------------------|------------------|-------------------|-----------------|
| **Random Forest** | **0.9960** | **8.90** | **6.91** | 37.47 | 0.0031 |
| XGBoost | 0.9957 | 9.25 | 7.33 | 1.57 | 0.0014 |
| Linear Regression | 0.4902 | 100.82 | 81.14 | 0.08 | 0.0039 |

### Best Model: Random Forest

**Key Metrics:**
- **R² Score:** 0.9960 (explains 99.6% of price variance)
- **RMSE:** 8.90 Lakhs (~₹890,000 average prediction error)
- **MAE:** 6.91 Lakhs (~₹691,000 median prediction error)
- **Overfitting:** Minimal (0.31% gap between train/test R²)

**Model Insights:**
- Random Forest significantly outperforms Linear Regression, capturing complex non-linear relationships
- Very low overfitting across all models indicates robust feature engineering
- XGBoost offers faster training (1.57s vs 37.47s) with comparable accuracy
- The selected Random Forest model achieves production-ready performance

---

## Preprocessing Pipeline Results

**Data Quality:**
- ✅ No missing values detected
- ✅ 0 outliers removed (0.00% of data)
- ✅ 8 categorical features encoded
- ✅ 2 high-cardinality features dropped (Amenities, Locality)

**Feature Engineering:**
- Created derived features: Price_per_SqFt, Age_of_Property
- Final feature set: 21 predictive variables
- Train/test split: 80/20 (200K/50K samples)

---

## Evaluation Metrics

Models are evaluated using:

- **RMSE (Root Mean Squared Error)** - Penalizes large errors
- **MAE (Mean Absolute Error)** - Average prediction error in Lakhs
- **R² Score** - Proportion of variance explained (0-1, higher is better)
- **Overfitting Gap** - Difference between training and test R² scores

---

## Key Findings

1. **Non-linear models dominate**: Random Forest and XGBoost achieve 99.6%+ R² scores, far exceeding Linear Regression's 49%
2. **Minimal overfitting**: All models show overfitting gaps under 0.4%, indicating robust preprocessing and feature engineering
3. **Production-ready accuracy**: Average prediction error of ₹6.91 lakhs on properties ranging from ₹10 lakhs to ₹500 lakhs
4. **Efficient training**: Despite 200K training samples, all models train in under 40 seconds
5. **Clean dataset**: Zero missing values and minimal outliers demonstrate high data quality

---

## Features Used for Prediction

**Numerical Features:**
- Size_in_SqFt, BHK, Year_Built, Floor_No, Total_Floors
- Age_of_Property, Nearby_Schools, Nearby_Hospitals
- Price_per_SqFt (derived)

**Categorical Features (Encoded):**
- State, City, Property_Type, Furnished_Status
- Public_Transport_Accessibility, Parking_Space
- Security, Facing, Owner_Type, Availability_Status

---

## API Endpoints

### `GET /`
Returns API information and available endpoints

### `GET /health`
Health check endpoint for monitoring

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-18T10:30:00"
}
```

### `POST /predict`
Make price predictions for properties

**Request Body:** JSON object with property features  
**Response:** Predicted price in Lakhs and INR with model metadata

---

## Deliverables

1. ✅ ML pipeline with full workflow (preprocess.py, train.py)
2. ✅ Cleaned dataset (250K samples, 21 features)
3. ✅ Trained Random Forest model (model.pkl)
4. ✅ Model comparison metrics (model_comparison.csv)
5. ✅ Flask REST API (http://localhost:8000)
6. ✅ Streamlit web interface (http://localhost:8501)
7. ✅ Comprehensive documentation with results

---

## Future Improvements

- Implement hyperparameter tuning (GridSearchCV/Optuna) for further optimization
- Add feature importance analysis and SHAP values for model interpretability
- Deploy to cloud platform (AWS/GCP/Azure) with production WSGI server
- Add real-time monitoring and logging
- Implement model versioning and A/B testing
- Create batch prediction endpoint for bulk pricing
- Add data drift detection and automated retraining

---

## System Requirements

**Minimum:**
- Python 3.9+
- 4GB RAM
- 2GB disk space

**Recommended:**
- Python 3.10+
- 8GB RAM
- SSD storage

---

## Troubleshooting

**Port already in use:**
```bash
# Change API port in api/app.py
app.run(host='0.0.0.0', port=8000)  # Change to 8001, 8002, etc.
```

**Model not found:**
```bash
# Ensure you've run the training pipeline first
python src/train.py
```

**Missing dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

---

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/ankushpanday1/india-house-price-prediction)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## License

This project is for educational purposes as part of a machine learning hackathon.

---

## Contact

For questions or issues, please open an issue in the repository.