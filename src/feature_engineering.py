"""
Feature Engineering Module
Creates derived features and performs feature selection
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones"""
    df_new = df.copy()
    
    # Price per square foot (if not already accurate)
    df_new['Price_per_SqFt_Calculated'] = (df_new['Price_in_Lakhs'] * 100000) / df_new['Size_in_SqFt']
    
    # Room density
    df_new['Room_Density'] = df_new['BHK'] / df_new['Size_in_SqFt'] * 1000
    
    # Floor ratio
    df_new['Floor_Ratio'] = df_new['Floor_No'] / df_new['Total_Floors']
    
    # Property age category
    df_new['Age_Category'] = pd.cut(df_new['Age_of_Property'], 
                                     bins=[0, 5, 10, 20, 35], 
                                     labels=['New', 'Recent', 'Old', 'Very_Old'])
    
    # Locality score (schools + hospitals)
    df_new['Locality_Score'] = df_new['Nearby_Schools'] + df_new['Nearby_Hospitals']
    
    # Size category
    df_new['Size_Category'] = pd.cut(df_new['Size_in_SqFt'],
                                      bins=[0, 1000, 2000, 3000, 5000],
                                      labels=['Small', 'Medium', 'Large', 'Very_Large'])
    
    print(f"✓ Created 7 new features")
    return df_new


def get_correlation_analysis(df: pd.DataFrame, target_col: str = 'Price_in_Lakhs'):
    """Analyze correlation with target variable"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numerical_cols:
        correlations = df[numerical_cols].corr()[target_col].sort_values(ascending=False)
        
        print("\nTop 10 Correlations with Price:")
        print("-" * 40)
        for feature, corr in correlations.head(10).items():
            print(f"{feature:30s}: {corr:6.3f}")
    
    return correlations


def prepare_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and prepare final feature set for modeling"""
    df_final = df.copy()
    
    # Drop unnecessary columns
    cols_to_drop = ['ID', 'Locality', 'Price_per_SqFt']
    df_final = df_final.drop(columns=cols_to_drop, errors='ignore')
    
    # One-hot encode remaining categorical variables
    categorical_cols = ['State', 'City', 'Property_Type', 'Amenities', 
                        'Facing', 'Owner_Type', 'Availability_Status',
                        'Age_Category', 'Size_Category']
    
    # Check which columns exist
    existing_cat_cols = [col for col in categorical_cols if col in df_final.columns]
    
    df_final = pd.get_dummies(df_final, columns=existing_cat_cols, drop_first=True)
    
    print(f"✓ Final feature count: {df_final.shape[1]}")
    return df_final


def save_engineered_data(df: pd.DataFrame, output_path: str):
    """Save feature-engineered data"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Feature-engineered data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Feature engineering for housing data')
    parser.add_argument('--input', type=str, default='data/processed/housing_cleaned.csv',
                        help='Input cleaned CSV file')
    parser.add_argument('--output', type=str, default='data/processed/housing_features.csv',
                        help='Output feature-engineered CSV file')
    args = parser.parse_args()
    
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load cleaned data
    print("\n[1/4] Loading cleaned data...")
    df = pd.read_csv(args.input)
    print(f"✓ Loaded: {df.shape}")
    
    # Create derived features
    print("\n[2/4] Creating derived features...")
    df = create_derived_features(df)
    print(f"✓ Current shape: {df.shape}")
    
    # Correlation analysis
    print("\n[3/4] Analyzing correlations...")
    correlations = get_correlation_analysis(df)
    
    # Prepare final features
    print("\n[4/4] Preparing final feature set...")
    df = prepare_final_features(df)
    print(f"✓ Final shape: {df.shape}")
    
    # Save
    print("\n" + "="*60)
    save_engineered_data(df, args.output)
    print("="*60)
    print("✓ Feature engineering completed successfully!")


if __name__ == '__main__':
    main()