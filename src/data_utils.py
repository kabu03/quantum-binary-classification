import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

def generate_two_moons():
    dataset_name = "two_moons"
    print(f"Processing dataset: {dataset_name}")
    X, y = make_moons(n_samples=1500, noise=0.25, random_state=42) # 1500 samples, with moderate noise
    two_moons_df = pd.DataFrame(np.c_[X, y], columns=['x1', 'x2', 'class'])
    return two_moons_df

def load_banknote():
    dataset_name = "banknote"
    print(f"Processing dataset: {dataset_name}")
    column_names = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
    df = pd.read_csv('data/data_banknote_authentication.txt', header=None, names=column_names)
    # Randomly shuffle the data to remove any order bias
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def clean_data(df):
    """
    Drops missing and duplicate rows. 
    In this dataset, we don't expect any missing or duplicate entries, but we add this step for robustness.
    """
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    
    if missing_count > 0 or duplicate_count > 0:
        print(f"Found {missing_count} missing values and {duplicate_count} duplicate rows. Dropping them!")
    
    df_clean = df.dropna().drop_duplicates()
    return df_clean

def split_data(df):
    """
    Splits the dataset into train, validation, and test sets with stratification to preserve class ratios.
    """
    X = df.drop('class', axis=1).values
    y = df['class'].values

    # First split: Train + Validation (80%) and Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Second split: Train (70%) and Validation (10%)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)
    # (0.125 × 0.8 = 0.10 overall for validation)

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """
    Scales features to have mean 0 and variance 1 using StandardScaler.
    Important: Fit the scaler only on the training data to prevent data leakage!
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def clean_split_scale(df):
    """
    Cleans the data, splits it into train/validation/test sets, and scales the features.
    """
    # Clean the data
    df_clean = clean_data(df)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
    
    # Scale the features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test