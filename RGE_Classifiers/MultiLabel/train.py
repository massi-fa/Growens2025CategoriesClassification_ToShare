import argparse
import os
import json
import time
import shutil
import ast

import torch
from math import sqrt
import pandas as pd
import numpy as np
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

from huggingface_hub import login

import matplotlib.pyplot as plt
import seaborn as sns

from torch.nn import BCEWithLogitsLoss

def multi_labels(row, categories):
    current_labels = ast.literal_eval(row['category_name'])
    current_labels = [modify_label(label) for label in current_labels]
    new_labels = []
    for cat in categories:
        if modify_label(cat) in current_labels:
            new_labels.append(1)
        else:
            new_labels.append(0)
    return new_labels


def split_filtered_dataset(df, target_col, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Splits the dataset, removing rows where the target column has value 0, and splits the remaining data into train/val/test."""
    assert train_size + val_size + test_size == 1, "The proportions must sum to 1."
    
    # Filter the DataFrame to keep only rows with at least one positive label (target == 1)
    df_filtered = df[df[target_col].apply(lambda x: sum(x)) > 0]
    
    # Split the filtered DataFrame into train, validation, and test sets
    train_df, temp_df = train_test_split(df_filtered, train_size=train_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, train_size=val_size / (val_size + test_size), random_state=random_state)
    
    return train_df, val_df, test_df


class BertDataset(Dataset):
    """Custom dataset for multilabel classification with BERT."""
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Args:
            texts (list of str): List of input texts.
            labels (list of list): List of binary label arrays for each text (multilabel).
            tokenizer (transformers.PreTrainedTokenizer): BERT tokenizer.
            max_length (int): Maximum length of the input sequences.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  
        label = torch.tensor(self.labels[idx], dtype=torch.float) 
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label  
        }


def calculate_token_distribution(df, text_column_name, tokenizer):
    """Calculates the token distribution for each row of the dataset."""
    token_counts = []
    for text in df[text_column_name]:
        tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True)
        token_counts.append(len(tokens))
    return token_counts


def save_token_distribution_percentile(token_counts, output_dir):
    """Saves the token distribution graph and percentiles as a JSON file."""
    percentiles = np.percentile(token_counts, [10, 25, 50, 75, 90])
    
    plt.figure(figsize=(8, 6))
    plt.hist(token_counts, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Token Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    token_distribution_path = os.path.join(output_dir, f'token_distribution.png')
    plt.savefig(token_distribution_path)
    plt.close()

    percentiles_dict = {
        "10th_percentile": percentiles[0],
        "25th_percentile": percentiles[1],
        "50th_percentile": percentiles[2],
        "75th_percentile": percentiles[3],
        "90th_percentile": percentiles[4]
    }
    percentiles_path = os.path.join(output_dir, f'token_percentiles.json')
    with open(percentiles_path, 'w') as f:
        json.dump(percentiles_dict, f, indent=4)

    print(f"Token distribution saved to: {token_distribution_path}")
    print(f"Percentiles saved to: {percentiles_path}")
    return token_distribution_path, percentiles_path


def compute_metrics(pred, threshold=0.5):
    """Calcola le metriche essenziali per la classificazione multilabel."""
    
    # Estrarre etichette vere e predette
    labels = pred.label_ids
    logits = pred.predictions
    
    # Applica sigmoid ai logits per ottenere probabilità
    probs = 1 / (1 + np.exp(-logits))
    
    # Convertire le probabilità in valori binari usando un threshold
    preds = (probs >= threshold).astype(int)
    
    # Dizionario per memorizzare le metriche
    metrics = {}
    
    # Accuracy generale
    metrics["accuracy"] = accuracy_score(labels, preds)
    
    # Calcolare macro-metrics (media tra tutte le classi)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    metrics["macro_precision"] = precision
    metrics["macro_recall"] = recall
    metrics["macro_f1"] = f1
    
    return metrics

def compute_metrics_test(test_results, num_labels, categories, output_dir=None, train_df=None, val_df=None, test_df=None):
    # Extract true and predicted labels
    labels = test_results.label_ids
    logits = test_results.predictions

    # Print min and max values of logits
    print(f"Min logit value: {np.min(logits)}")
    print(f"Max logit value: {np.max(logits)}")

    # Apply sigmoid to logits to get probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Print min and max values of probabilities
    print(f"Min probability value: {np.min(probs)}")
    print(f"Max probability value: {np.max(probs)}")

    # Dictionary to store all metrics
    all_metrics = {}
    best_threshold = 0.5
    best_f1 = 0

    # Add data distribution information if available
    if train_df is not None and val_df is not None and test_df is not None:
        train_counts = count_instances(train_df, 'labels', categories)
        val_counts = count_instances(val_df, 'labels', categories)
        test_counts = count_instances(test_df, 'labels', categories)

        # Create a dictionary with distribution statistics
        data_distribution = {}
        for category in categories:
            cat_name = modify_label(category)
            data_distribution[cat_name] = {
                "train_positive": train_counts[cat_name]['positive'],
                "train_negative": train_counts[cat_name]['negative'],
                "val_positive": val_counts[cat_name]['positive'],
                "val_negative": val_counts[cat_name]['negative'],
                "test_positive": test_counts[cat_name]['positive'],
                "test_negative": test_counts[cat_name]['negative'],
                "total_positive": train_counts[cat_name]['positive'] + val_counts[cat_name]['positive'] + test_counts[cat_name]['positive'],
                "total_negative": train_counts[cat_name]['negative'] + val_counts[cat_name]['negative'] + test_counts[cat_name]['negative'],
                "imbalance_ratio": (train_counts[cat_name]['negative'] + val_counts[cat_name]['negative'] + test_counts[cat_name]['negative']) /
                                  max(1, (train_counts[cat_name]['positive'] + val_counts[cat_name]['positive'] + test_counts[cat_name]['positive']))
            }

        # Add data distribution to metrics
        all_metrics["data_distribution"] = data_distribution

        # Print a summary of the distribution
        print("\n=== Class Distribution Summary ===")
        print(f"{'Category':<30} {'Train Pos':<10} {'Train Neg':<10} {'Val Pos':<10} {'Val Neg':<10} {'Test Pos':<10} {'Test Neg':<10} {'Imbalance':<10}")
        print("-" * 100)
        for category in categories:
            cat_name = modify_label(category)
            cat_dist = data_distribution[cat_name]
            print(f"{category:<30} {cat_dist['train_positive']:<10} {cat_dist['train_negative']:<10} {cat_dist['val_positive']:<10} {cat_dist['val_negative']:<10} {cat_dist['test_positive']:<10} {cat_dist['test_negative']:<10} {cat_dist['imbalance_ratio']:.2f}")

    # Only use the classic threshold 0.5
    threshold = 0.5
    preds_bin = (probs >= threshold).astype(int)

    # Check if all predictions are 0 or all are 1
    all_zeros = np.all(preds_bin == 0)
    all_ones = np.all(preds_bin == 1)

    if all_zeros:
        print(f"Threshold {threshold}: All predictions are 0")
    if all_ones:
        print(f"Threshold {threshold}: All predictions are 1")

    # Metrics for threshold 0.5
    threshold_metrics = {}

    # Overall accuracy
    threshold_metrics["accuracy"] = float(accuracy_score(labels, preds_bin))

    # Calculate macro-metrics (average across all classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds_bin, average='macro', zero_division=0
    )
    threshold_metrics["macro_precision"] = float(precision_macro)
    threshold_metrics["macro_recall"] = float(recall_macro)
    threshold_metrics["macro_f1"] = float(f1_macro)

    # Calculate micro-metrics (aggregate all TP, FP, FN)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds_bin, average='micro', zero_division=0
    )
    threshold_metrics["micro_precision"] = float(precision_micro)
    threshold_metrics["micro_recall"] = float(recall_micro)
    threshold_metrics["micro_f1"] = float(f1_micro)

    # Metrics for each class
    class_metrics = {}
    for i, category in enumerate(categories):
        cat_name = modify_label(category)

        # Confusion matrix for this class
        cm = confusion_matrix(labels[:, i], preds_bin[:, i])

        # Handle different shapes of confusion matrix
        if cm.size == 1:  # Only one value (all predictions are the same)
            if np.all(labels[:, i] == 0) and np.all(preds_bin[:, i] == 0):
                # All true negatives
                tn, fp, fn, tp = cm[0][0], 0, 0, 0
            elif np.all(labels[:, i] == 1) and np.all(preds_bin[:, i] == 1):
                # All true positives
                tn, fp, fn, tp = 0, 0, 0, cm[0][0]
            elif np.all(labels[:, i] == 0) and np.all(preds_bin[:, i] == 1):
                # All false positives
                tn, fp, fn, tp = 0, cm[0][0], 0, 0
            elif np.all(labels[:, i] == 1) and np.all(preds_bin[:, i] == 0):
                # All false negatives
                tn, fp, fn, tp = 0, 0, cm[0][0], 0
            else:
                # Default case (shouldn't happen)
                tn, fp, fn, tp = 0, 0, 0, 0
        elif cm.shape == (1, 1):
            # Another possible shape for a single value
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        elif cm.shape == (2, 2):
            # Normal case with 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle any other unexpected shape
            print(f"Warning: Unexpected confusion matrix shape {cm.shape} for category {category}")
            tn, fp, fn, tp = 0, 0, 0, 0

        # Accuracy for this class
        class_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        # Precision, recall, f1 for this class (positive class)
        precision_pos, recall_pos, f1_pos, support_pos = precision_recall_fscore_support(
            labels[:, i], preds_bin[:, i], pos_label=1, average='binary', zero_division=0
        )

        # Precision, recall, f1 for this class (negative class)
        precision_neg, recall_neg, f1_neg, support_neg = precision_recall_fscore_support(
            labels[:, i], preds_bin[:, i], pos_label=0, average='binary', zero_division=0
        )

        # Micro and macro metrics for this specific category
        precision_macro_cat, recall_macro_cat, f1_macro_cat, _ = precision_recall_fscore_support(
            labels[:, i], preds_bin[:, i], average='macro', zero_division=0
        )

        precision_micro_cat, recall_micro_cat, f1_micro_cat, _ = precision_recall_fscore_support(
            labels[:, i], preds_bin[:, i], average='micro', zero_division=0
        )

        class_metrics[cat_name] = {
            # Add accuracy for this class
            "accuracy": float(class_accuracy),
            # Metrics for the positive class
            "positive": {
                "precision": float(precision_pos),
                "recall": float(recall_pos),
                "f1": float(f1_pos),
                "support": int(tp + fn)  # Total number of positive examples
            },
            # Metrics for the negative class
            "negative": {
                "precision": float(precision_neg),
                "recall": float(recall_neg),
                "f1": float(f1_neg),
                "support": int(tn + fp)  # Total number of negative examples
            },
            # Macro metrics for this category
            "macro": {
                "precision": float(precision_macro_cat),
                "recall": float(recall_macro_cat),
                "f1": float(f1_macro_cat)
            },
            # Micro metrics for this category
            "micro": {
                "precision": float(precision_micro_cat),
                "recall": float(recall_micro_cat),
                "f1": float(f1_micro_cat)
            },
            # Confusion matrix
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn)
            }
        }

    threshold_metrics["class_metrics"] = class_metrics
    all_metrics[f"threshold_{threshold}"] = threshold_metrics

    # Add information about the best threshold (always 0.5 here)
    all_metrics["best_threshold"] = {
        "value": best_threshold,
        "macro_f1": best_f1
    }

    # Save all metrics to a JSON file if an output directory is specified
    if output_dir:
        metrics_path = os.path.join(output_dir, 'detailed_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Detailed metrics saved to: {metrics_path}")

    return all_metrics


def count_instances(df, target_col, categories):
    """Counts the number of positive and negative instances for each label (class) in the binary array."""
    # Initialize a dictionary to store the count of labels for each class
    label_counts = {modify_label(category): {'positive': 0, 'negative': 0} for category in categories}

    # For each example in the dataframe, count positives and negatives for each class
    for index, row in df.iterrows():
        labels = row[target_col]  # Get the binary array of labels
        for i, label in enumerate(labels):
            category = modify_label(categories[i])  # Name of the class corresponding to the position
            if label == 1:
                label_counts[category]['positive'] += 1
            else:
                label_counts[category]['negative'] += 1

    return label_counts

def save_class_distribution(train_df, val_df, test_df, output_dir, target_col, categories):
    """Saves the class distribution (positive and negative counts per label) as a JSON file."""
    train_label_counts = count_instances(train_df, target_col, categories)
    val_label_counts = count_instances(val_df, target_col, categories)
    test_label_counts = count_instances(test_df, target_col, categories)

    # Build the dictionary with the label distribution for each dataset split
    class_distribution = {
        'train': train_label_counts,
        'val': val_label_counts,
        'test': test_label_counts
    }

    # Save the distribution as a JSON file
    class_distribution_json_path = os.path.join(output_dir, f'class_distribution.json')
    with open(class_distribution_json_path, 'w') as f:
        json.dump(class_distribution, f, indent=4)

    print(f"Class distribution saved to: {class_distribution_json_path}")
    return class_distribution_json_path

def sanitize_filename(name):
    """
    Create a safe filename by removing or replacing invalid characters.
    This function is only used for filenames and does not affect label logic.
    """
    # Replace problematic characters
    import re
    # Remove invalid characters from filenames
    invalid_chars = r'[\\/*?:"<>|]'
    sanitized = re.sub(invalid_chars, "_", name)

    # Limit the length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]

    return sanitized

def plot_and_save_confusion_matrix(test_results, output_dir, categories, threshold=0.5):
    """Plots and saves the confusion matrix for each label in a multilabel classification task."""
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract true (y_true) and predicted (y_pred) values from predictions
    y_true = test_results.label_ids
    y_pred = test_results.predictions

    # Ensure y_pred contains only binary values (0 or 1)
    y_pred_bin = (y_pred >= threshold).astype(int)

    # Create a list for the paths of the confusion matrices
    confusion_matrix_paths = []

    # Calculate and save the confusion matrix for each category
    for i, category in enumerate(categories):
        try:
            # Calculate the confusion matrix for each category separately
            cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])  # For class 'i'

            # Create the confusion matrix visualization
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for {category}')

            # Use modify_label for consistency with the rest of the code
            category_filename = modify_label(category)
            # Further sanitize to ensure a valid filename
            safe_filename = sanitize_filename(category_filename)

            # Debug info
            print(f"Category: '{category}'")
            print(f"Modified label: '{category_filename}'")
            print(f"Safe filename: '{safe_filename}'")

            # Save the confusion matrix as an image
            confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{safe_filename}.png')

            # Try to save with error handling
            try:
                plt.savefig(confusion_matrix_path)
                plt.close()
                print(f"Confusion matrix for {category} saved at {confusion_matrix_path}")
                confusion_matrix_paths.append(confusion_matrix_path)
            except Exception as e:
                print(f"Error saving confusion matrix for {category}: {e}")
                # Try an alternative path with a simple numeric filename
                alt_path = os.path.join(output_dir, f'confusion_matrix_category_{i}.png')
                try:
                    plt.savefig(alt_path)
                    plt.close()
                    print(f"Confusion matrix for {category} saved at alternative path: {alt_path}")
                    confusion_matrix_paths.append(alt_path)
                except Exception as e2:
                    print(f"Failed to save confusion matrix even to alternative path: {e2}")
                    plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {category}: {e}")

    # Return a list of the paths to the confusion matrix images
    return confusion_matrix_paths


def save_metrics(metrics, output_dir):
    """Saves the metrics as a JSON file."""
    metrics_path = os.path.join(output_dir, f'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    return metrics_path

def create_sample_category_dict(y_true, categories):
    # Create a list for the dictionaries of individual samples
    sample_dicts = []

    for i in range(len(y_true)):  # Loop over each sample
        sample_dict = {}
        for j, category in enumerate(categories):  # Loop over each category
            # Assign the binary value for that category and convert to int
            sample_dict[category] = int(y_true[i, j])  # Convert to int
        sample_dicts.append(sample_dict)

    return sample_dicts

def modify_label(label):
    return label.replace(" / ", "_").replace(" & ", "_and_").replace(" ", "")


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

    # Read categories from categoriesList.json in the current directory
    categories_path = os.path.join(os.path.dirname(__file__), 'categoriesList.json')
    with open(categories_path, 'r') as f:
        categories = json.load(f)

    if args.categories_number == 0:
        print("Run with all categories")
    else:
        categories = categories[:args.categories_number]
    print('Processing', len(categories), ':', categories)

    print('Using model:', args.model_name)

    best_model = args.load_best_model_at_end
    print(f"Load best model is: {best_model}")

    analisys_save_folder_path = args.analisys_save_folder_path
    os.makedirs(analisys_save_folder_path, exist_ok=True)


    # Initialize tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")


    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(categories),
        problem_type='multi_label_classification',
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("📥 Loading pre-split datasets with pandas...")
    train_df = pd.read_csv(args.train_file_path).dropna()
    val_df = pd.read_csv(args.val_file_path).dropna()
    test_df = pd.read_csv(args.test_file_path).dropna()
    print("✅ Datasets loaded successfully.")

    full_dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print("Full dataset shape:", full_dataset.shape)
    # Calculate and save token distribution
    token_counts = calculate_token_distribution(full_dataset, args.text_column_name, tokenizer)
    token_distribution_path, percentiles_path = save_token_distribution_percentile(token_counts, analisys_save_folder_path)
    
    # Apply multi_labels to each split
    print("MultiLabeling the train, validation, and test datasets...")
    train_df['labels'] = train_df.apply(lambda row: multi_labels(row, categories), axis=1)
    val_df['labels'] = val_df.apply(lambda row: multi_labels(row, categories), axis=1)
    test_df['labels'] = test_df.apply(lambda row: multi_labels(row, categories), axis=1)
    print("✅ Datasets multilabeled.")
    print("Train DataFrame shape:", train_df.shape)
    print("Validation DataFrame shape:", val_df.shape)
    print("Test DataFrame shape:", test_df.shape)

    # Class distribution
    class_distribution_json_path = save_class_distribution(train_df, val_df, test_df, analisys_save_folder_path, 'labels', categories)



    # Create datasets for the Trainer
    train_dataset = BertDataset(train_df[args.text_column_name].tolist(), train_df['labels'].tolist(), tokenizer, args.max_length)
    val_dataset = BertDataset(val_df[args.text_column_name].tolist(), val_df['labels'].tolist(), tokenizer, args.max_length)
    test_dataset = BertDataset(test_df[args.text_column_name].tolist(), test_df['labels'].tolist(), tokenizer, args.max_length)

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, num_labels=len(categories), categories=categories, threshold=args.threshold),
    )

    # Start training
    trainer.train()
    print("✅ Training completed.")

    # Save the model and tokenizer locally using Hugging Face's save_pretrained        
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
        
    

    # Evaluation on the test set
    test_results = trainer.predict(test_dataset)
        
    detailed_metrics = compute_metrics_test(
        test_results,
        num_labels=len(categories),
        categories=categories,
        output_dir=analisys_save_folder_path,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )
        
    # Upload detailed metrics to S3
    detailed_metrics_path = os.path.join(analisys_save_folder_path, 'detailed_metrics.json')
        
    # Confusion matrix
    # Calculate and save the confusion matrix for each label in the multilabel classification task
    confusion_matrix_paths = plot_and_save_confusion_matrix(test_results, analisys_save_folder_path, categories, threshold=args.threshold)


    # Check the lengths of the arrays
    y_true = test_results.label_ids
    y_pred = test_results.predictions

    y_pred_bin = (y_pred >= args.threshold).astype(int)

    y_true = create_sample_category_dict(y_true, categories)

    y_pred_bin = create_sample_category_dict(y_pred_bin, categories)

    # Create the DataFrame with the correct data
    results_df = pd.DataFrame({
        "post_identifier": test_df['post_identifier'],
        "text": test_df['structured_text'],
        "true_label": y_true,
        "predicted_label": y_pred_bin
    })

    results_csv_path = os.path.join(analisys_save_folder_path, "test_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)

if __name__ == '__main__':
    main()