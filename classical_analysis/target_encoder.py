import pandas as pd

class TargetEncoder:
    def __init__(self, smoothing=1.0, col_delimiters=None):
        """
        Initialize the target encoder.
        
        Parameters:
        - smoothing: Smoothing parameter for target encoding.
        - delimiter: Delimiter used to separate multiple categories.
        """
        self.smoothing = smoothing
        self.col_delimiters = col_delimiters
        self.global_mean = None        # Global target mean from training data.
        self.encoding_map = {}         # Mapping for each column.
        self.fitted_columns = []       # List of columns that were fitted.

    def fit(self, X, target, columns):
        """
        Fit the encoder on the training data. This function calculates the 
        statistics and encoding mapping for each specified column independently.
        
        Parameters:
        - X: pandas DataFrame that includes the feature columns and target.
        - target: The name of the target column.
        - columns: A list (or a single string) of column names to encode.
        
        Returns:
        - self (fitted encoder)
        """
        if isinstance(columns, str):
            columns = [columns]
        self.fitted_columns = columns
        self.global_mean = X[target].mean()
        
        # For each column, calculate stats and the smoothed encoding mapping.
        for col in columns:
            stats = {}  # Temporary storage for count and sum of target per category for this column.
            for _, row in X.iterrows():
                val = row[col]
                t = row[target]
                # Check if the value is a multiple category entry.
                if isinstance(val, str) and col in self.col_delimiters:
                    categories = [v.strip() for v in val.split(self.col_delimiters[col])]
                else:
                    categories = [str(val).strip()]
                    
                for cat in categories:
                    if cat not in stats:
                        stats[cat] = {'count': 0, 'sum': 0.0}
                    stats[cat]['count'] += 1
                    stats[cat]['sum'] += t
            
            # Compute the smoothed encoding for each category in this column.
            self.encoding_map[col] = {}
            for cat, stat in stats.items():
                count = stat['count']
                cat_mean = stat['sum'] / count if count > 0 else 0
                self.encoding_map[col][cat] = (count * cat_mean + self.smoothing * self.global_mean) / (count + self.smoothing)
                
        return self

    def transform(self, X, columns=None):
        """
        Transform the dataset using the mapping learned during fit.
        Each column is processed independently, and if a category was not seen during training,
        the global mean is used as a fallback.
        
        Parameters:
        - X: pandas DataFrame to transform.
        - columns: (Optional) list of columns to transform. If not provided, the fitted columns are used.
                   
        Returns:
        - A new DataFrame with additional encoded columns.
        """
        if columns is None:
            columns = self.fitted_columns
        
        df = X.copy()
        for col in columns:
            if col not in self.encoding_map:
                raise ValueError(f"Column '{col}' was not fitted in the encoder!")
            encoded_values = []
            for _, row in df.iterrows():
                val = row[col]
                # For multiple category entries, compute the average encoding.
                if isinstance(val, str) and col in self.col_delimiters:
                    categories = [v.strip() for v in val.split(self.col_delimiters[col])]
                    encodings = [self.encoding_map[col].get(cat, self.global_mean) for cat in categories]
                    encoded = sum(encodings) / len(encodings) if encodings else self.global_mean
                else:
                    cat = str(val).strip()
                    encoded = self.encoding_map[col].get(cat, self.global_mean)
                encoded_values.append(encoded)
            df[col] = encoded_values
        return df

    def fit_transform(self, X, target, columns):
        """
        Fit the encoder on the data and then transform it.
        
        Parameters:
        - X: pandas DataFrame.
        - target: The name of the target column.
        - columns: A list (or single string) of column names to encode.
        
        Returns:
        - Transformed DataFrame with additional encoded columns.
        """
        return self.fit(X, target, columns).transform(X)

if __name__ == "__main__": 
    # -----------------------------
    # Example Usage with Train and Test DataFrames
    # -----------------------------
    # Training DataFrame with independent columns.
    train_data = {
        'single_prior': ['A', 'B', 'A', 'C', 'B', 'A'],
        'multiple_priors': ['A;B', 'B;C', 'A', 'C;D', 'B;D', 'A;D'],
        'target': [1, 0, 1, 0, 0, 1]
    }
    train_df = pd.DataFrame(train_data)

    # Test DataFrame with potentially unseen categories.
    test_data = {
        'single_prior': ['A', 'B', 'D'],  # 'D' is unseen in training for this column.
        'multiple_priors': ['A;C', 'B;E', 'D;E']  # 'E' and 'D' might be unseen.
    }
    test_df = pd.DataFrame(test_data)

    # Instantiate the encoder.
    encoder = TargetEncoder(smoothing=1.0, col_delimiters={'multiple_prios': ';'})

    # Fit the encoder on the training data.
    encoder.fit(train_df, target='target', columns=['single_prior', 'multiple_priors'])

    # Transform both training and test datasets.
    train_encoded = encoder.transform(train_df)
    test_encoded = encoder.transform(test_df)

    print("Encoded Training DataFrame:")
    print(train_encoded)
    print("\nEncoded Test DataFrame:")
    print(test_encoded)
