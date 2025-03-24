import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch
from captum.attr import IntegratedGradients

import data_preprocessing as dp

# ----------------------------
# Hyperparameters
# ----------------------------
learning_rate = 2e-05
train_batch_size = 16
eval_batch_size = 16
seed_val = 42
num_epochs = 5
lr_scheduler_type = "linear"

set_seed(seed_val)

# ----------------------------
# Define Metrics Function
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='macro', zero_division=0)
    rec = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def create_results_directory(base_path="llm_analysis/results/classification"):
    """Create the directory structure for saving results"""
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def save_window_results(base_dir, window_index, metrics, trainer):
    """Save all results for a specific window"""
    window_dir = base_dir / f"window_{window_index}"
    window_dir.mkdir(exist_ok=True)
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(window_dir / 'metrics.csv', index=False)
    
    # Save model
    trainer.save_model(window_dir / 'final_model')

if __name__ == '__main__':

    # df = dp.load_dataset('data/updated_dataset.parquet.gzip')

    # df = df.sample(100, random_state=42)  # Test
    df = pd.read_parquet("data/updated_dataset_preprocessed.parquet.gzip")
    window_df = dp.create_rolling_windows(df, date_col=dp.TIME_COL, window_size=2, window_step=6)

    # ----------------------------
    # Process Each Window and Train/Evaluate the Model
    # ----------------------------
    results_list = []

    # Create results directory
    results_dir = create_results_directory()

    # Assume window_df is a list/array of DataFrames, one for each time window.
    # Each DataFrame must contain the categorical columns and a target column dp.TARGET_COL
    for window_index, window in enumerate(window_df):
        if window_index != 10:
            continue # Only process the 10 windows for now
        
        print(f"\nProcessing Window {window_index+1} (from {window[dp.TIME_COL].min().date()} to {window[dp.TIME_COL].max().date()}):")
        
        # ----------------------------
        # Load Pre-trained Model & Tokenizer
        # ----------------------------
        model_name = "neuralmind/bert-base-portuguese-cased"  # Change if necessary
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Print label distribution for the whole window
        print("\nWindow Label Distribution:")
        print(window[dp.TARGET_COL].value_counts())
        print(f"Total samples in window: {len(window)}")
        print("-" * 30)
        
        # Make a copy to avoid modifying the original
        window = window.copy()
        
        # Create a new "text" column by concatenating the categorical columns
        window["text"] = window[dp.CATEGORICAL_COLS].astype(str).apply(lambda x: " ".join(x), axis=1)

        # Ensure the target column 'label' exists
        if dp.TARGET_COL not in window.columns:
            raise ValueError("Target column 'label' not found in the DataFrame for window {}".format(window_index))
        
        # Split data: 70% train, 30% test (shuffled)
        train_df, test_df = train_test_split(window, test_size=0.3, random_state=seed_val, shuffle=True)
        
        # Convert pandas DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
        
        # Tokenize the datasets using the "text" column
        def tokenize_function(example):
            return tokenizer(example["text"], truncation=True, padding='max_length', max_length=512)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", dp.TARGET_COL])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", dp.TARGET_COL])
        
        # Define training arguments for this window
        output_dir = str(results_dir / f"window_{window_index}")
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            logging_steps=50,
            seed=seed_val,
            lr_scheduler_type=lr_scheduler_type,
            save_strategy="epoch",
            logging_dir=str(results_dir / f"window_{window_index}" / "logs")
        )
        
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train the model on the current window
        trainer.train()
        
        # Evaluate on the test set
        metrics = trainer.evaluate()
        
        metrics["window"] = window_index  # Track window identifier
        results_list.append(metrics)
        
        # Save results using the new methodology
        save_window_results(results_dir, window_index, metrics, trainer)
        
        # Print metrics for current window
        print(f"\nMetrics per Window {window_index}:")
        print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"Precision: {metrics['eval_precision']:.4f}")
        print(f"Recall: {metrics['eval_recall']:.4f}")
        print(f"F1 Score: {metrics['eval_f1']:.4f}")
        print("-" * 50)
        
        # ----------------------------
        # Integrated Gradients Attribution for a Sample Test Instance
        # ----------------------------
        print("\n=== Starting Integrated Gradients Analysis ===")
        print("This analysis helps understand which parts of the input text most influenced the model's decision")
        
        # Select a sample from the test dataset
        sample = test_dataset[0]
        input_ids_sample = sample["input_ids"].unsqueeze(0)
        attention_mask_sample = sample["attention_mask"].unsqueeze(0)
        sample_label = sample[dp.TARGET_COL]
        
        print(f"\nAnalyzing sample with true label: {sample_label}")
        print("Preparing input embeddings...")
        
        # Obtain embeddings for the sample using the model's embedding layer
        embedding_layer = model.get_input_embeddings()
        embeddings = embedding_layer(input_ids_sample)
        print(f"Input embedding shape: {embeddings.shape}")

        # Define a forward function that takes embeddings as input
        def forward_func(input_embeds):
            outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask_sample)
            return outputs.logits

        print("\nInitializing Integrated Gradients...")
        # Create IntegratedGradients instance with the forward function
        ig = IntegratedGradients(forward_func)
        
        # Use a baseline of zero embeddings (same shape as embeddings)
        baseline = torch.zeros_like(embeddings)
        print("Using zero embeddings as baseline for attribution")
        
        print("\nComputing attributions...")
        # Compute Integrated Gradients attributions
        attributions, delta = ig.attribute(embeddings, baseline, target=sample_label, return_convergence_delta=True)
        print(f"Convergence delta: {delta.item():.6f}")
        
        # Sum attributions across embedding dimensions
        attributions_sum = attributions.sum(dim=-1).squeeze(0)
        
        # Convert input ids to tokens for readability
        tokens = tokenizer.convert_ids_to_tokens(input_ids_sample[0])
        
        print("\n=== Token Attribution Analysis ===")
        print("Positive values indicate tokens that pushed towards the prediction")
        print("Negative values indicate tokens that pushed against the prediction")
        print("-" * 50)
        
        # Find max attribution for scaling
        max_attr = max(abs(score) for score in attributions_sum.tolist())
        for token, score in zip(tokens, attributions_sum.tolist()):
            # Scale the score for better visualization
            scaled_score = score / max_attr
            # Add direction indicators
            direction = "→" if score > 0 else "←" if score < 0 else "•"
            print(f"Token: {token:15} | Attribution: {score:>8.4f} {direction} | Scaled: {scaled_score:>8.4f}")
        
        print("\n=== Analysis Complete ===")
        print(f"Total number of tokens analyzed: {len(tokens)}")
        print(f"Average attribution magnitude: {attributions_sum.abs().mean().item():.4f}")
        print("-" * 50)
        
    # Save final results summary
    print("\nSaving final results...")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(results_dir / "results_summary_classification.csv", index=False)
    print(f"Results successfully saved to {results_dir}")
