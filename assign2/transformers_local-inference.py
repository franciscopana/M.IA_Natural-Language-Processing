import os
import json
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Custom dataset for text classification inference"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def find_csv_file():
    """Find the only CSV file in the current directory"""
    csv_files = glob.glob("*.csv")
    
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV file found in the current directory")
    elif len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found: {csv_files}. Please ensure only one CSV file is present.")
    
    return csv_files[0]


def find_model_directory():
    """Find the inference model directory"""
    model_dirs = glob.glob("inference_model*/**", recursive=True)
    model_dirs = [d for d in model_dirs if os.path.isdir(d) and 'config.json' in os.listdir(d)]
    
    if len(model_dirs) == 0:
        raise FileNotFoundError("No inference model directory found. Looking for directories starting with 'inference_model' containing model files.")
    elif len(model_dirs) > 1:
        print(f"Multiple model directories found: {model_dirs}")
        print(f"Using the first one: {model_dirs[0]}")
    
    return model_dirs[0]


def load_model_and_tokenizer(model_path):
    """Load the trained model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load results.json for label mappings
    results_path = os.path.join(model_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        label2id = results.get("label2id", {})
        id2label = results.get("id2label", {})
        # Convert string keys back to integers for id2label
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        print("Warning: results.json not found. Label mappings might not be available.")
        label2id = {}
        id2label = {}
    
    return model, tokenizer, label2id, id2label


def preprocess_dataset(csv_path):
    """Load and preprocess the CSV dataset"""
    print(f"Loading dataset from: {csv_path}")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Remove 'source' and 'reasoning' columns if they exist
    columns_to_drop = ['source', 'reasoning']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        print(f"Dropping columns: {existing_columns_to_drop}")
        df = df.drop(columns=existing_columns_to_drop)
    
    print(f"Dataset shape after dropping columns: {df.shape}")
    print(f"Remaining columns: {list(df.columns)}")
    
    # Check for required columns
    if 'text' not in df.columns:
        raise ValueError("'text' column not found in dataset")
    if 'label' not in df.columns:
        raise ValueError("'label' column not found in dataset")
    
    # Remove any rows with missing text or labels
    initial_length = len(df)
    df = df.dropna(subset=['text', 'label'])
    final_length = len(df)
    
    if initial_length != final_length:
        print(f"Removed {initial_length - final_length} rows with missing text or labels")
    
    return df


def run_inference(model, tokenizer, texts, batch_size=32, device='cpu'):
    """Run inference on the text data"""
    print(f"Running inference on {len(texts)} samples using device: {device}")
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predicted class
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)
    
    return np.array(predictions)


def evaluate_predictions(true_labels, predictions, label2id, id2label):
    """Calculate metrics for the predictions"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Convert string labels to numeric if necessary
    if isinstance(true_labels[0], str):
        true_labels_numeric = [label2id.get(label, -1) for label in true_labels]
        # Check if any labels weren't found in mapping
        unknown_labels = [label for label, numeric in zip(true_labels, true_labels_numeric) if numeric == -1]
        if unknown_labels:
            print(f"Warning: Unknown labels found: {set(unknown_labels)}")
            # Filter out unknown labels
            valid_indices = [i for i, numeric in enumerate(true_labels_numeric) if numeric != -1]
            true_labels_numeric = [true_labels_numeric[i] for i in valid_indices]
            predictions = [predictions[i] for i in valid_indices]
            print(f"Filtered dataset to {len(valid_indices)} samples with known labels")
    else:
        true_labels_numeric = true_labels
    
    true_labels_numeric = np.array(true_labels_numeric)
    predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels_numeric, predictions)
    f1_weighted = f1_score(true_labels_numeric, predictions, average='weighted')
    f1_macro = f1_score(true_labels_numeric, predictions, average='macro')
    f1_micro = f1_score(true_labels_numeric, predictions, average='micro')
    
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels_numeric, predictions, average='weighted'
    )
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    
    # Print classification report
    if id2label:
        target_names = [id2label.get(i, f"Class_{i}") for i in range(len(np.unique(true_labels_numeric)))]
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels_numeric, predictions, target_names=target_names))
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'num_samples': len(true_labels_numeric)
    }


def save_results(metrics, output_file="inference_results.json"):
    """Save the evaluation metrics to a JSON file"""
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Results saved successfully!")


def main():
    print("Starting local inference script...")
    
    # Set device
    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Find and load dataset
        csv_file = find_csv_file()
        df = preprocess_dataset(csv_file)
        
        # Find and load model
        model_path = find_model_directory()
        model, tokenizer, label2id, id2label = load_model_and_tokenizer(model_path)
        
        # Extract texts and labels
        texts = df['text'].tolist()
        true_labels = df['label'].tolist()
        
        # Run inference
        predictions = run_inference(model, tokenizer, texts, device=device)
        
        # Evaluate predictions
        metrics = evaluate_predictions(true_labels, predictions, label2id, id2label)
        
        # Save results
        save_results(metrics)
        
        print(f"\nInference completed successfully!")
        print(f"F1 Score: {metrics['f1_weighted']:.4f}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()