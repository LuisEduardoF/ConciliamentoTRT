import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.ensemble import RandomForestClassifier
import data_preprocessing as dp

# ----------------------------
# Hyperparameters
# ----------------------------
seed_val = 42
train_batch_size = 16
eval_batch_size = 16
set_seed(seed_val)

# ----------------------------
# Define Metrics Function
# ----------------------------
def compute_metrics_from_labels(true_labels, predictions):
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, average='macro', zero_division=0)
    rec = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

if __name__ == '__main__':
    print("Loading dataset...")
    df = pd.read_parquet("data/updated_dataset_preprocessed.parquet.gzip")
    print(f"Dataset loaded with shape: {df.shape}")

    print("\nCreating rolling windows...")
    window_df = dp.create_rolling_windows(df, date_col=dp.TIME_COL, window_size=2, window_step=6)
    print(f"Created {len(window_df)} windows")

    print("\nLoading BERT model and tokenizer...")
    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        print("Model moved to GPU")
    print("Model loaded successfully")

    results_list = []

    for window_index, window in enumerate(window_df):
        
        print(f"\nProcessing Window {window_index+1} (from {window[dp.TIME_COL].min().date()} to {window[dp.TIME_COL].max().date()}):")
        
        # Print label distribution for the whole window
        print("\nWindow Label Distribution:")
        print(window[dp.TARGET_COL].value_counts())
        print(f"Total samples in window: {len(window)}")
        print("-" * 30)
        
        # Make a copy to avoid modifying the original DataFrame
        window = window.copy()
        
        # Create a new "text" column by concatenating the categorical columns
        window["text"] = window[dp.CATEGORICAL_COLS].astype(str).apply(lambda x: " ".join(x), axis=1)
        
        # Ensure the target column exists
        if dp.TARGET_COL not in window.columns:
            raise ValueError(f"Target column '{dp.TARGET_COL}' not found in the DataFrame for window {window_index}")
        
        print(f"\nSplitting window {window_index+1} into train/test sets...")
        train_df, test_df = train_test_split(window, test_size=0.3, random_state=seed_val, shuffle=True)
        print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
        
        # ----------------------------
        # Tokenize the "text" column for train and test sets
        # ----------------------------
        print("\nTokenizing texts...")
        def tokenize_texts(df):
            return tokenizer(list(df["text"]), truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        
        train_tokens = tokenize_texts(train_df)
        test_tokens = tokenize_texts(test_df)
        print("Tokenization completed")
        
        # Move tokens to GPU if available
        if torch.cuda.is_available():
            train_tokens = {k: v.to("cuda") for k, v in train_tokens.items()}
            test_tokens = {k: v.to("cuda") for k, v in test_tokens.items()}
            print("Tokens moved to GPU")
        
        # ----------------------------
        # Extract Embeddings Using BERT
        # ----------------------------
        print("\nExtracting BERT embeddings...")
        def get_embeddings(tokens, batch_size):
            embeddings = []
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            dataset_size = input_ids.shape[0]
            with torch.no_grad():
                for i in range(0, dataset_size, batch_size):
                    batch_input_ids = input_ids[i:i+batch_size]
                    batch_attention_mask = attention_mask[i:i+batch_size]
                    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    # Use the [CLS] token representation from the last hidden state
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(batch_embeddings)
            embeddings = np.concatenate(embeddings, axis=0)
            return embeddings

        train_embeddings = get_embeddings(train_tokens, train_batch_size)
        print(f"Train embeddings shape: {train_embeddings.shape}")
        test_embeddings = get_embeddings(test_tokens, eval_batch_size)
        print(f"Test embeddings shape: {test_embeddings.shape}")
        
        # ----------------------------
        # Prepare Labels and Train Random Forest
        # ----------------------------
        print("\nTraining Random Forest classifier...")
        y_train = train_df[dp.TARGET_COL].values
        y_test = test_df[dp.TARGET_COL].values
        
        clf = RandomForestClassifier(random_state=seed_val)
        clf.fit(train_embeddings, y_train)
        print("Random Forest training completed")
        
        print("\nMaking predictions...")
        y_pred = clf.predict(test_embeddings)
        
        # Compute metrics
        metrics = compute_metrics_from_labels(y_test, y_pred)
        metrics["window"] = window_index
        results_list.append(metrics)
        
        # Print metrics for current window
        print(f"\nMetrics for Window {window_index}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("-" * 50)

    # ----------------------------
    # Save All Results to CSV
    print("\nSaving final results...")
    # ----------------------------
    results_df = pd.DataFrame(results_list)
    results_csv_path = "results_summary_random_forest.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results successfully saved to {results_csv_path}")
