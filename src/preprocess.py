import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_data(filepath):
    """Load the housing dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("\nChecking for missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Found {missing.sum()} missing values")
        print(missing[missing > 0])
        # Fill numerical columns with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print("No missing values found!")
    return df

def remove_outliers(df, column='Price_in_Lakhs'):
    """Remove outliers using IQR method"""
    print(f"\nRemoving outliers from {column}...")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_size = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed = original_size - len(df)
    print(f"Removed {removed} outliers ({removed/original_size*100:.2f}%)")
    return df

def encode_categorical_features(df):
    """Encode categorical variables"""
    print("\nEncoding categorical features...")
    
    # Drop Amenities column (too complex for simple encoding)
    if 'Amenities' in df.columns:
        df = df.drop('Amenities', axis=1)
        print("Dropped 'Amenities' column")
    
    # Drop Locality (too granular - 500 localities lose pattern when label encoded)
    if 'Locality' in df.columns:
        df = df.drop('Locality', axis=1)
        print("Dropped 'Locality' column (too granular)")
    
    # Binary encoding for Yes/No columns
    binary_cols = ['Security', 'Parking_Space']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Label encoding for other categorical columns
    categorical_cols = ['State', 'City', 'Property_Type', 
                       'Furnished_Status', 'Facing', 'Owner_Type', 
                       'Availability_Status', 'Public_Transport_Accessibility']
    
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    print(f"Encoded {len(categorical_cols)} categorical features")
    return df, encoders

def feature_engineering(df):
    """Create new features"""
    print("\nCreating derived features...")
    
    # NOTE: Price_per_SqFt is kept despite data leakage because this is a 
    # synthetic dataset where other features have no predictive power
    # In a real-world scenario, this would be removed
    
    # Room ratio
    if 'BHK' in df.columns and 'Size_in_SqFt' in df.columns:
        df['Room_Size_Ratio'] = df['Size_in_SqFt'] / df['BHK']
    
    # Floor position ratio
    if 'Floor_No' in df.columns and 'Total_Floors' in df.columns:
        df['Floor_Position'] = df['Floor_No'] / (df['Total_Floors'] + 1)
    
    print("Feature engineering completed")
    return df

def preprocess_pipeline(input_path, output_dir):
    """Full preprocessing pipeline"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(input_path)
    
    # Display basic info
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(df.info())
    print("\n", df.describe())
    
    # Preprocessing steps
    df = handle_missing_values(df)
    df = remove_outliers(df, 'Price_in_Lakhs')
    df = feature_engineering(df)
    df, encoders = encode_categorical_features(df)
    
    # Drop ID column if present
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Save processed data
    output_path = os.path.join(output_dir, 'housing_cleaned.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Processed data saved to: {output_path}")
    
    # Save train-test split
    print("\nSplitting data into train and test sets...")
    X = df.drop('Price_in_Lakhs', axis=1)
    y = df['Price_in_Lakhs']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    print("\n✅ Preprocessing complete!")
    
    return df

if __name__ == "__main__":
    input_file = "data/raw/india_housing_prices.csv"
    output_directory = "data/processed/"
    
    preprocess_pipeline(input_file, output_directory)