# Proprietary Code – © 2025 Growens, Inc.
# All rights reserved.
#
# This source code is proprietary and confidential.
# Unauthorized copying, distribution, modification, or use
# of this code, in whole or in part, is strictly prohibited
# without prior written permission from the owner.

import argparse
import os
import json
import time
import shutil
import ast

import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import re

def modify_label(label):
    return label.replace(" / ", "_or_").replace(" & ", "_and_").replace(" ", "_").strip()

def restore_label(modified_label):
    """Restore the original format of a modified label."""
    return modified_label.replace("_and_", " & ").replace("_or_", "/").replace("_", " ")

def create_binary_dataset(row, category):
    """Binarizes the row based on the presence of the category."""
    current_labels = ast.literal_eval(row['category_name'])
    current_labels = [modify_label(label) for label in current_labels]
    if modify_label(category) in current_labels:
        return 1
    else:
        return 0


def split_imbalanced_dataset(df, target_col, train_size=0.7, val_size=0.15, test_size=0.15, balance=1, random_state=42):
    """Splits the imbalanced dataset into train/val/test, balancing the negative classes based on the 'balance' parameter."""
    assert train_size + val_size + test_size == 1, "The proportions must sum to 1."
    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]
    num_pos = len(df_pos)
    num_neg = int(num_pos * balance)
    df_neg_balanced = df_neg.sample(n=num_neg, random_state=random_state)
    train_pos, temp_pos = train_test_split(df_pos, train_size=train_size, random_state=random_state)
    val_pos, test_pos = train_test_split(temp_pos, train_size=val_size / (val_size + test_size), random_state=random_state)
    train_neg, temp_neg = train_test_split(df_neg_balanced, train_size=train_size, random_state=random_state)
    val_neg, test_neg = train_test_split(temp_neg, train_size=val_size / (val_size + test_size), random_state=random_state)
    train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, val_df, test_df

def sanitize_filename(name):
    """
    Create a safe file name by removing or replacing invalid characters.
    This function is used only for file names and does not affect the labeling logic.
    """
    # Replace problematic characters
    # Remove invalid characters in file names
    invalid_chars = r'[\\/*?:"<>|]'
    sanitized = re.sub(invalid_chars, "_", name)
    # Limit the length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

class BertDataset(Dataset):
    """Custom dataset for training with BERT."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def calculate_token_distribution(df, text_column_name, tokenizer):
    """Calculates the token distribution for each row of the dataset."""
    token_counts = []
    for text in df[text_column_name]:
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True)
        token_counts.append(len(tokens))
    return token_counts


def save_token_distribution_percentile(token_counts, output_dir, category):
    """Saves the token distribution graph and percentiles as a JSON file."""
    percentiles = np.percentile(token_counts, [10, 25, 50, 75, 90])
    
    plt.figure(figsize=(8, 6))
    plt.hist(token_counts, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Token Distribution for {category}')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    token_distribution_path = os.path.join(output_dir, f'token_distribution_{category}.png')
    plt.savefig(token_distribution_path)
    plt.close()

    percentiles_dict = {
        "category": category,
        "10th_percentile": percentiles[0],
        "25th_percentile": percentiles[1],
        "50th_percentile": percentiles[2],
        "75th_percentile": percentiles[3],
        "90th_percentile": percentiles[4]
    }
    percentiles_path = os.path.join(output_dir, f'token_percentiles_{category}.json')
    with open(percentiles_path, 'w') as f:
        json.dump(percentiles_dict, f, indent=4)

    print(f"Token distribution saved to: {token_distribution_path}")
    print(f"Percentiles saved to: {percentiles_path}")
    return token_distribution_path, percentiles_path


def compute_metrics(pred, threshold=0.5):
    """Calculate essential metrics for binary classification using threshold."""
    # Extract true labels
    labels = pred.label_ids
    # Convert logits to probabilities and apply threshold
    if len(pred.predictions.shape) > 1 and pred.predictions.shape[1] == 2:
        print(f"🔍 Predictions with shape {pred.predictions.shape} - using softmax")
        probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=1).numpy()
        preds = (probs[:, 1] >= threshold).astype(int)  # Positive class with threshold
    else:
        print(f"🔍 Predictions with shape {pred.predictions.shape} - using direct threshold")
        preds = (pred.predictions >= threshold).astype(int)

    # Dictionary to store metrics
    metrics = {}
    # Overall accuracy
    metrics["accuracy"] = accuracy_score(labels, preds)
    # Calculate macro-metrics (average across all classes)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    metrics["macro_precision"] = precision
    metrics["macro_recall"] = recall
    metrics["macro_f1"] = f1
    return metrics

def compute_metrics_test(test_results, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7], output_dir=None, category=None):
    """
    Computes metrics for binary classification for threshold 0.5 only.
    
    Args:
        test_results: Prediction results with label_ids and predictions
        output_dir: Directory where to save the results
        category: Category name for file naming
    Returns:
        Dict: Dictionary with computed metrics for threshold 0.5
    """
    # Extract true labels
    labels = test_results.label_ids

    # Handle predictions for binary classification
    if len(test_results.predictions.shape) > 1 and test_results.predictions.shape[1] == 2:
        # For binary classification, we have logits for 2 classes
        # Convert to probabilities for the positive class (class 1)
        print(f"🔍 Test predictions with shape {test_results.predictions.shape} - using softmax")
        probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=1).numpy()
        preds = probs[:, 1]  # Take the probability of the positive class
    else:
        # If predictions are already probabilities or single values
        print(f"🔍 Test predictions with shape {test_results.predictions.shape} - using direct values")
        preds = test_results.predictions

    threshold = 0.5
    # Convert predictions to binary values using threshold 0.5
    preds_bin = (preds >= threshold).astype(int)

    # Dictionary to store all metrics
    all_metrics = {}

    # Dictionary for metrics at threshold 0.5
    threshold_metrics = {}

    # Overall accuracy
    threshold_metrics["accuracy"] = float(accuracy_score(labels, preds_bin))

    # Precision, Recall, and F1-score for the positive class
    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        labels, preds_bin, pos_label=1, average='binary', zero_division=0
    )

    # Precision, Recall, and F1-score for the negative class
    precision_neg, recall_neg, f1_neg, _ = precision_recall_fscore_support(
        labels, preds_bin, pos_label=0, average='binary', zero_division=0
    )

    # Macro and Micro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds_bin, average='macro', zero_division=0
    )

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds_bin, average='micro', zero_division=0
    )

    threshold_metrics["macro"] = {
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_score": float(f1_macro)
    }

    threshold_metrics["micro"] = {
        "precision": float(precision_micro),
        "recall": float(recall_micro),
        "f1_score": float(f1_micro)
    }

    threshold_metrics["positive"] = {
        "precision": float(precision_pos),
        "recall": float(recall_pos),
        "f1_score": float(f1_pos)
    }

    threshold_metrics["negative"] = {
        "precision": float(precision_neg),
        "recall": float(recall_neg),
        "f1_score": float(f1_neg)
    }

    # Confusion matrix
    cm = confusion_matrix(labels, preds_bin)
    tn, fp, fn, tp = cm.ravel()
    threshold_metrics["confusion_matrix"] = {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }

    all_metrics[f"threshold_{threshold}"] = threshold_metrics

    # Add best threshold info (always 0.5 in this case)
    all_metrics["best_threshold"] = {
        "value": threshold,
        "macro_f1": f1_macro
    }

    # Save all metrics to a JSON file if output_dir is specified
    if output_dir:
        metrics_path = os.path.join(output_dir, f'detailed_metrics_{category}.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Binary classification metrics saved to: {metrics_path}")

    return all_metrics


def flatten_metrics(metrics_dict, prefix=""):
    """Flattens a nested metrics dictionary."""
    flattened = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix + key + "_"))
        else:
            flattened[prefix + key] = float(value)
    return flattened


def count_instances(df):
    """Counts the positive and negative instances."""
    pos = df[df['label'] == 1].shape[0]
    neg = df[df['label'] == 0].shape[0]
    return pos, neg


def save_class_distribution(train_df, val_df, test_df, output_dir, category):
    """Saves the class distribution as a JSON file."""
    train_pos, train_neg = count_instances(train_df)
    val_pos, val_neg = count_instances(val_df)
    test_pos, test_neg = count_instances(test_df)

    class_distribution = {
        'train': {'positive': train_pos, 'negative': train_neg, 'total': train_pos + train_neg},
        'val': {'positive': val_pos, 'negative': val_neg, 'total': val_pos + val_neg},
        'test': {'positive': test_pos, 'negative': test_neg, 'total': test_pos + test_neg}
    }
    class_distribution_json_path = os.path.join(output_dir, f'class_distribution_{modify_label(category)}.json')
    with open(class_distribution_json_path, 'w') as f:
        json.dump(class_distribution, f, indent=4)
    
    print(f"Class distribution saved to: {class_distribution_json_path}")
    return class_distribution_json_path


def plot_and_save_confusion_matrix(test_results, output_dir, category):
    """Plots and saves the confusion matrix."""
    y_true = test_results.label_ids
    y_pred = test_results.predictions.argmax(-1)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {category}')
    
    # Usa modify_label per la coerenza con il resto del codice
    category_filename = modify_label(category)
    # Poi sanitizza ulteriormente per garantire un nome file valido
    safe_filename = sanitize_filename(category_filename)

    confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{safe_filename}.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    print(f"Confusion matrix saved at {confusion_matrix_path}")
    return confusion_matrix_path


def save_metrics(metrics, output_dir, category):
    """Saves the metrics as a JSON file."""
    metrics_path = os.path.join(output_dir, f'metrics_{modify_label(category)}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    return metrics_path

def main():

    # Always read experiment_configuration.json from the current directory
    config_path = os.path.join(os.path.dirname(__file__), 'experiment_configuration.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Convert config dict to an object for attribute-style access
    class ArgsObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = ArgsObj(config)

    print("Arguments and their values (from experiment_configuration.json):")
    for arg, value in config.items():
        print(f"{arg}: {value}")

    analisys_save_folder_path = args.analisys_save_folder_path
    os.makedirs(analisys_save_folder_path, exist_ok=True)


    # Initialize tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Initialize tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2, 
        problem_type='single_label_classification',
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load pre-split datasets directly with pandas
    print("📥 Loading pre-split datasets with pandas...")
    train_df = pd.read_csv(args.train_file_path).dropna()
    val_df = pd.read_csv(args.val_file_path).dropna()
    test_df = pd.read_csv(args.test_file_path).dropna()
    print("✅ Datasets loaded successfully.")

    # Binarize the datasets
    print("🧪 Binarizing the datasets...")
    train_df['label'] = train_df.apply(lambda row: create_binary_dataset(row, args.category), axis=1)
    val_df['label'] = val_df.apply(lambda row: create_binary_dataset(row, args.category), axis=1)
    test_df['label'] = test_df.apply(lambda row: create_binary_dataset(row, args.category), axis=1)
    print("✅ Binarization completed.")

    full_dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Calculate and save token distribution
    token_counts = calculate_token_distribution(full_dataset, args.text_column_name, tokenizer)
    token_distribution_path, percentiles_path = save_token_distribution_percentile(token_counts, analisys_save_folder_path, args.category)

        # Create datasets for the Trainer
    train_dataset = BertDataset(train_df[args.text_column_name].tolist(), train_df['label'].tolist(), tokenizer, args.max_length)
    val_dataset = BertDataset(val_df[args.text_column_name].tolist(), val_df['label'].tolist(), tokenizer, args.max_length)

        # Class distribution
    class_distribution_json_path = save_class_distribution(train_df, val_df, test_df, analisys_save_folder_path, args.category)
        
    # Training Configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        logging_strategy=args.logging_strategy,
        seed=args.seed,
        #report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, threshold=args.threshold)  # Use threshold of 0.5 during training
    )

    # Start training
    trainer.train()
    print("✅ Training completed.")

    # Save the model and tokenizer locally using Hugging Face's save_pretrained
    model_name = "final_model_" + sanitize_filename(modify_label(args.category))
    final_model_path = os.path.join(args.output_dir, model_name)
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ Model and tokenizer saved locally in: {final_model_path}")

    

    # Evaluation on the test set
    test_dataset = BertDataset(test_df[args.text_column_name].tolist(), test_df['label'].tolist(), tokenizer, args.max_length)
    test_results = trainer.predict(test_dataset)

    detailed_metrics = compute_metrics_test(
        test_results,
        output_dir=analisys_save_folder_path,
        category=args.category
    )
        
        
    detailed_metrics_path = os.path.join(analisys_save_folder_path, f'detailed_metrics_{args.category}.json')
    
    # Confusion matrix
    confusion_matrix_path = plot_and_save_confusion_matrix(test_results, analisys_save_folder_path, args.category)
    
    # Save predictions
    results_df = pd.DataFrame({
        "post_identifier": test_df['post_identifier'],
        "text": test_df['structured_text'],
        "true_label": test_results.label_ids,
        "predicted_label": test_results.predictions.argmax(-1)
    })
    results_csv_path = os.path.join(analisys_save_folder_path, "test_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)


if __name__ == '__main__':
    main()