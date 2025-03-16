import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import joblib

import data_preprocessing as dp
import target_encoder as te

def create_rolling_windows(df, date_col, window_size, window_step):
    """
    Split the DataFrame into rolling windows based on a datetime column.
    The window size is interpreted in years and the window step in months.

    Parameters:
    - df: pandas DataFrame containing the datetime column.
    - date_col: string, name of the datetime column.
    - window_size: integer, window size in years.
    - window_step: integer, step between windows in months.
    
    Returns:
    - List of DataFrame slices corresponding to each rolling window.
    """
    # Ensure the date_col is datetime type.
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    window_size_delta = relativedelta(years=window_size)
    window_step_delta = relativedelta(months=window_step)
    
    windows = []
    start_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # Create windows until start_date goes beyond the maximum date.
    while start_date <= max_date:
        end_date = start_date + window_size_delta
        window_df = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
        if not window_df.empty:
            windows.append(window_df)
        start_date = start_date + window_step_delta
    
    return windows

def run_classification_on_split(X_train, X_test, y_train, y_test, random_state=42):
    """
    Train three classification models on provided train/test splits and print evaluation metrics.
    """
    # Print label distribution for train and test sets
    print("\nLabel Distribution:")
    print("Training set:")
    print(pd.Series(y_train).value_counts())
    print("\nTesting set:")
    print(pd.Series(y_test).value_counts())
    print("-" * 30)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=random_state),
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state)
    }
    
    results = {}
    predictions = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        predictions[name] = y_pred
        
        # Print detailed classification report with zero_division parameter
        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 30)
        
        # Store results in dictionary with zero_division parameter
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        results[name] = {
            'accuracy': report['accuracy'],
            'macro_avg_precision': report['macro avg']['precision'],
            'macro_avg_recall': report['macro avg']['recall'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_precision': report['weighted avg']['precision'],
            'weighted_avg_recall': report['weighted avg']['recall'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }
        
    return results, predictions

def save_window_data(window_num, X_train, X_test, y_train, y_test, predictions, fitted_models):
    """
    Save window data and models to corresponding directories.
    """
    # Create window directory
    window_dir = f'classical_analysis/results/window_{window_num}'
    os.makedirs(window_dir, exist_ok=True)
    
    # Save data
    X_train.to_csv(f'{window_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{window_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{window_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{window_dir}/y_test.csv', index=False)
    
    # Save predictions for each model
    for model_name, y_pred in predictions.items():
        np.save(f'{window_dir}/y_pred_{model_name}.npy', y_pred)
    
    # Save models
    for model_name, model in fitted_models.items():
        joblib.dump(model, f'{window_dir}/{model_name}.joblib')

# -----------------------------
# Example Usage in Main
# -----------------------------
if __name__ == '__main__':
    
    # Load dataset (assumed to be in a parquet file).
    df = dp.load_dataset('data/updated_dataset.parquet.gzip')
    
    print("\nDataset after loading:")
    print(df.head(10))
    
    # Create rolling windows based on a date column.
    windows = create_rolling_windows(df, date_col=dp.TIME_COL, window_size=2, window_step=6)
    print(f"\nCreated {len(windows)} rolling windows.")
    
    # Initialize results tracking dataframe
    results_all = []
    
    # For each rolling window, perform a train/test split and evaluate models.
    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i+1} (from {window[dp.TIME_COL].min().date()} to {window[dp.TIME_COL].max().date()}):")
        
        # Print label distribution for the whole window
        print("\nWindow Label Distribution:")
        print(window[dp.TARGET_COL].value_counts())
        print(f"Total samples in window: {len(window)}")
        print("-" * 30)
        
        # Split the window into training and testing sets.
        train, test = train_test_split(window, test_size=0.3, random_state=42)
        
        # Apply target encoding: Fit on train and transform train and test.
        print("Applying Target Encoding...")
        encoder = te.TargetEncoder(col_delimiters=dp.COL_DELIMITERS)
        train_encoded = encoder.fit_transform(train, target=dp.TARGET_COL, columns=dp.CATEGORICAL_COLS)
        test_encoded = encoder.transform(test)
        
        # Define features and target.
        # (Assuming your feature columns are 'feature1' and 'feature2'; adjust as needed.)
        X_train = train_encoded[dp.CATEGORICAL_COLS]
        y_train = train_encoded[dp.TARGET_COL]
        X_test = test_encoded[dp.CATEGORICAL_COLS]
        y_test = test_encoded[dp.TARGET_COL]
        
        print(f"Evaluating models for Window {i+1}:")
        results, predictions = run_classification_on_split(X_train, X_test, y_train, y_test)
        
        # Save window data and fitted models
        fitted_models = {
            'LogisticRegression': LogisticRegression(random_state=42).fit(X_train, y_train),
            'RandomForest': RandomForestClassifier(random_state=42).fit(X_train, y_train),
            'GradientBoosting': GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
        }
        save_window_data(i+1, X_train, X_test, y_train, y_test, predictions, fitted_models)
        
        # Add window information to results
        for model_name, model_results in results.items():
            window_results = {
                'window_number': i+1,
                'start_date': window[dp.TIME_COL].min().date(),
                'end_date': window[dp.TIME_COL].max().date(),
                'model': model_name,
                **model_results
            }
            results_all.append(window_results)
        
        
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results_all)
    output_path = 'classical_analysis/results/window_classification_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
