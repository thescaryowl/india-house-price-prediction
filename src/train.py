import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import time

def load_train_test_data(processed_dir):
    """Load preprocessed train and test data"""
    print("Loading train and test data...")
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    
    X_train = train_df.drop('Price_in_Lakhs', axis=1)
    y_train = train_df['Price_in_Lakhs']
    X_test = test_df.drop('Price_in_Lakhs', axis=1)
    y_test = test_df['Price_in_Lakhs']
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance on both train and test sets"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}...")
    print('='*60)
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test set predictions
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"TRAINING SET:")
    print(f"  RMSE: {train_rmse:.2f} Lakhs")
    print(f"  MAE:  {train_mae:.2f} Lakhs")
    print(f"  R² Score: {train_r2:.4f}")
    
    print(f"\nTEST SET:")
    print(f"  RMSE: {test_rmse:.2f} Lakhs")
    print(f"  MAE:  {test_mae:.2f} Lakhs")
    print(f"  R² Score: {test_r2:.4f}")
    
    # Calculate overfitting metrics
    overfitting = train_r2 - test_r2
    print(f"\nOverfitting Gap (Train R² - Test R²): {overfitting:.4f}")
    
    return {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'overfitting_gap': overfitting
    }

def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train Linear Regression model"""
    print("\n Training Linear Regression...")
    start_time = time.time()
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, "Linear Regression")
    metrics['training_time'] = training_time
    
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("\n Training Random Forest...")
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, "Random Forest")
    metrics['training_time'] = training_time
    
    return model, metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model"""
    print("\n Training XGBoost...")
    start_time = time.time()
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, "XGBoost")
    metrics['training_time'] = training_time
    
    return model, metrics

def save_best_model(models, metrics, model_dir):
    """Save the best performing model based on test R² score"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Find best model by test R² score
    best_idx = max(range(len(metrics)), key=lambda i: metrics[i]['test_r2'])
    best_model = models[best_idx]
    best_metrics = metrics[best_idx]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_metrics['model_name']}")
    print(f"{'='*60}")
    print(f"Test R² Score: {best_metrics['test_r2']:.4f}")
    print(f"Test RMSE: {best_metrics['test_rmse']:.2f} Lakhs")
    print(f"Test MAE: {best_metrics['test_mae']:.2f} Lakhs")
    print(f"Training Time: {best_metrics['training_time']:.2f} seconds")
    
    # Save model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n Best model saved to: {model_path}")
    
    # Save metrics comparison
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(model_dir, 'model_comparison.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f" Metrics saved to: {metrics_path}")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print('='*60)
    print(metrics_df[['model_name', 'test_r2', 'test_rmse', 'test_mae', 'overfitting_gap', 'training_time']].to_string(index=False))
    
    return best_model, best_metrics

def train_all_models(processed_dir='data/processed/', model_dir='models/'):
    """Train all models and save the best one"""
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_train_test_data(processed_dir)
    
    # Train models
    models = []
    metrics = []
    
    # Linear Regression
    lr_model, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
    models.append(lr_model)
    metrics.append(lr_metrics)
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    models.append(rf_model)
    metrics.append(rf_metrics)
    
    # XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    models.append(xgb_model)
    metrics.append(xgb_metrics)
    
    # Save best model
    best_model, best_metrics = save_best_model(models, metrics, model_dir)
    
    print("\n" + "="*60)
    print(" TRAINING PIPELINE COMPLETED!")
    print("="*60)
    
    return best_model, metrics

if __name__ == "__main__":
    train_all_models()