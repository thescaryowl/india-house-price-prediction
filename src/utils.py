import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_model(model_path):
    """Load a saved model"""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_model(model, model_path):
    """Save a model to disk"""
    try:
        joblib.dump(model, model_path)
        print(f"Model saved successfully to {model_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def plot_predictions(y_true, y_pred, title="Actual vs Predicted Prices"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (Lakhs)')
    plt.ylabel('Predicted Price (Lakhs)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('models/predictions_plot.png')
    plt.close()
    print("Prediction plot saved to models/predictions_plot.png")

def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """Plot residuals"""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price (Lakhs)')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('models/residuals_plot.png')
    plt.close()
    print("Residual plot saved to models/residuals_plot.png")

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        plt.close()
        print("Feature importance plot saved to models/feature_importance.png")
    else:
        print("Model does not have feature_importances_ attribute")

def get_model_metrics(y_true, y_pred):
    """Calculate and return model metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }