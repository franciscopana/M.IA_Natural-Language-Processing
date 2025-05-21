import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)
import torch
from kaggle_secrets import UserSecretsClient


# ------------------
# Setup secrets
# ------------------
secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
hf_token = secrets.get_secret("huggingface")
os.environ["HUGGINGFACE_TOKEN"] = hf_token


# ------------------
# Configuration
# ------------------
@dataclass
class Config:
    model_name: str = "roberta-base"
    dataset_name: str = "Intel/polite-guard"
    cl_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    wandb_project: str = "domain-adaptation"
    mlm_epochs: int = 3
    mlm_learning_rate: float = 2e-5
    enable_domain_adapt: bool = True
    mlm_validation_split: float = 0.1

    # these will be set in __post_init__:
    run_name: str = None
    output_dir: str = None
    mlm_output_dir: str = None

    def __post_init__(self):
        timestamp = datetime.now().strftime("%B-%d_%H-%M")
        self.run_name = f"{timestamp}_{self.model_name}_lr-{self.learning_rate}_bs-{self.batch_size}_epochs-{self.cl_epochs}"
        self.output_dir = f"./training_output/{self.model_name}/{self.run_name}"
        self.mlm_output_dir = f"./mlm_output/{self.model_name}/{self.run_name}"


# ------------------
# Data Loading & Preprocessing
# ------------------
def load_and_preprocess(cfg: Config):
    print(f"Loading dataset {cfg.dataset_name}")
    raw = load_dataset(cfg.dataset_name)
    ds = DatasetDict({
        "train": raw["train"],
        "validation": raw["validation"],
        "test": raw["test"],
    })

    label_list = sorted(raw["train"].unique("label"))
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 1. Classification tokenization:
    def tokenize_classification(batch):
        outputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        outputs["labels"] = [label2id[l] for l in batch["label"]]
        return outputs

    tok_labeled = ds.map(
        tokenize_classification,
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    # 2. MLM tokenization with validation split
    def tokenize_mlm(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_special_tokens_mask=True,
        )

    # Create train/val split for MLM
    if cfg.mlm_validation_split > 0:
        mlm_splits = raw["train"].train_test_split(
            test_size=cfg.mlm_validation_split, 
            seed=42
        )
        tok_unlabeled_train = mlm_splits["train"].map(
            tokenize_mlm,
            batched=True,
            remove_columns=raw["train"].column_names,
        )
        tok_unlabeled_val = mlm_splits["test"].map(
            tokenize_mlm,
            batched=True,
            remove_columns=raw["train"].column_names,
        )
        tok_unlabeled = {"train": tok_unlabeled_train, "validation": tok_unlabeled_val}
    else:
        tok_unlabeled = {"train": raw["train"].map(
            tokenize_mlm,
            batched=True,
            remove_columns=raw["train"].column_names,
        )}

    return tok_labeled, tok_unlabeled, tokenizer, label2id


# ------------------
# Model & Metrics
# ------------------
def build_model_and_metrics(model_path: str, num_labels: int, tokenizer):
    print(f"Loading model from {model_path}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        print("Loaded pretrained classification model")
    except:
        print("Converting MLM model to classification model")
        mlm_model = AutoModelForMaskedLM.from_pretrained(model_path)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            _from_model=mlm_model,
            ignore_mismatched_sizes=True
        )
        del mlm_model
    
    model.resize_token_embeddings(len(tokenizer))

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return model, compute_metrics

def plot_and_save_confusion_matrix(cm: np.ndarray, labels: List[str], path: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            f"{cm[i, j]}",
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


# ------------------
# Domain Adaptation
# ------------------
def perform_domain_adaptation(cfg: Config, tok_unlabeled_data, tokenizer):
    print("Starting domain adaptation (MLM) training")
    
    mlm_model = AutoModelForMaskedLM.from_pretrained(cfg.model_name)
    mlm_model.resize_token_embeddings(len(tokenizer))
    
    mlm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )
    mlm_training_args = TrainingArguments(
        output_dir=cfg.mlm_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.mlm_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=cfg.mlm_learning_rate,
        weight_decay=cfg.weight_decay,
        report_to="wandb",
        load_best_model_at_end=True if "validation" in tok_unlabeled_data else False,
        eval_strategy="epoch" if "validation" in tok_unlabeled_data else "no",
        save_strategy="epoch" if "validation" in tok_unlabeled_data else "steps",
    )
    
    # Set up trainer with validation if available
    mlm_trainer_kwargs = {
        "model": mlm_model,
        "args": mlm_training_args,
        "train_dataset": tok_unlabeled_data["train"],
        "data_collator": mlm_data_collator,
    }
    
    if "validation" in tok_unlabeled_data:
        mlm_trainer_kwargs["eval_dataset"] = tok_unlabeled_data["validation"]
    
    mlm_trainer = Trainer(**mlm_trainer_kwargs)
    
    def compute_perplexity(model, eval_dataset, batch_size=8):
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            collate_fn=mlm_data_collator
        )
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            
            batch_loss = outputs.loss.item() * batch["attention_mask"].sum().item()
            total_loss += batch_loss
            total_tokens += batch["attention_mask"].sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return perplexity
    
    print("Starting MLM training")
    mlm_trainer.train()
    
    if "validation" in tok_unlabeled_data:
        try:
            final_perplexity = compute_perplexity(
                mlm_model, 
                tok_unlabeled_data["validation"],
                batch_size=cfg.batch_size
            )
            wandb.log({"final_mlm_perplexity": final_perplexity})
            print(f"Final MLM perplexity: {final_perplexity}")
        except Exception as e:
            print(f"Could not compute perplexity: {e}")
    
    # Save the model
    mlm_trainer.save_model(cfg.mlm_output_dir)
    tokenizer.save_pretrained(cfg.mlm_output_dir)
    print(f"Domain adaptation complete. Model saved to {cfg.mlm_output_dir}")
    
    return cfg.mlm_output_dir


# ------------------
# Main training pipeline
# ------------------
def main():
    cfg = Config()

    # Initialize W&B
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=vars(cfg),
        save_code=True,
    )

    # Load & preprocess
    tok_labeled_data, tok_unlabeled_data, tokenizer, label2id = load_and_preprocess(cfg)
    num_labels = len(label2id)

    # --- Domain Adaptation (Masked Language Model) ---
    if cfg.enable_domain_adapt:
        adapted_model_path = perform_domain_adaptation(cfg, tok_unlabeled_data, tokenizer)
    else:
        adapted_model_path = cfg.model_name
        print("Skipping domain adaptation")

    # --- Text Classification ---
    # Build classification model from the domain-adapted model
    cl_model, compute_metrics = build_model_and_metrics(adapted_model_path, num_labels, tokenizer)
    
    # Training
    cl_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    cl_training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.cl_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        report_to="wandb",
    )
    cl_trainer = Trainer(
        model=cl_model,
        args=cl_training_args,
        train_dataset=tok_labeled_data["train"],
        eval_dataset=tok_labeled_data["validation"],
        data_collator=cl_data_collator,
        compute_metrics=compute_metrics,
    )
    print("Starting classification training")
    cl_trainer.train()

    # Evaluate
    print("Evaluating on test set")
    pred_output = cl_trainer.predict(tok_labeled_data["test"])
    cm = confusion_matrix(pred_output.label_ids, np.argmax(pred_output.predictions, axis=-1))

    # Save model & results
    print("Saving model and results")
    cl_trainer.save_model()
    os.makedirs(cfg.output_dir, exist_ok=True)

    results = {
        "metrics": pred_output.metrics,
        "confusion_matrix": cm.tolist(),
        "label2id": label2id,
        "domain_adapted": cfg.enable_domain_adapt,
    }
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plot & log confusion matrix
    cm_path = os.path.join(cfg.output_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, list(label2id.keys()), cm_path)
    wandb.log({"confusion_matrix": wandb.Image(cm_path)})

    # Log misclassifications
    texts = load_dataset(cfg.dataset_name)["test"]["text"]
    table = wandb.Table(columns=["text", "true", "pred"])

    for i, logit in enumerate(pred_output.predictions):
        true = pred_output.label_ids[i]
        pred = np.argmax(logit)
        if true != pred:
            table.add_data(texts[i], true, pred)

    wandb.log({"misclassifications": table})

    test_metrics = pred_output.metrics
    metrics_table = wandb.Table(columns=["metric", "value"])
    for name, val in test_metrics.items():
        metrics_table.add_data(name, val)
    wandb.log({"test_metrics_table": metrics_table})

    last_eval = [h for h in cl_trainer.state.log_history if h.get("eval_loss")][-1]
    eval_table = wandb.Table(columns=["metric", "value"])
    for k, v in last_eval.items():
        if k.startswith("eval_"):
            eval_table.add_data(k, v)
    wandb.log({"eval_metrics_table": eval_table})

    wandb.log({
        "roc": wandb.plot.roc_curve(
            pred_output.label_ids, 
            pred_output.predictions, 
            labels=list(label2id.keys())
        ),
        "pr": wandb.plot.pr_curve(
            pred_output.label_ids, 
            pred_output.predictions, 
            labels=list(label2id.keys())
        ),
    })

    wandb.finish()


if __name__ == "__main__":
    main()
