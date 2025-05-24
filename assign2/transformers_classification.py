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
    DataCollatorForLanguageModeling,
    set_seed
)
import torch
import torch.nn.functional as F
from kaggle_secrets import UserSecretsClient


# ------------------
# Setup secrets
# ------------------
try:
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    hf_token = secrets.get_secret("huggingface")
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
except:
    print("Kaggle secrets not found. Ensure WANDB_API_KEY and HUGGINGFACE_TOKEN are set in your environment if not on Kaggle.")
    if not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set.")
    if not os.environ.get("HUGGINGFACE_TOKEN"):
        print("Warning: HUGGINGFACE_TOKEN not set.")


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
    wandb_project: str = "domain-adaptation-seeded"
    mlm_epochs: int = 2
    mlm_learning_rate: float = 2e-5
    mlm_probability: float = 0.1
    enable_domain_adapt: bool = True
    mlm_validation_split: float = 0.1
    seed: int = 42

    # these will be set in __post_init__:
    run_name: str = None
    output_dir: str = None
    mlm_output_dir: str = None

    def __post_init__(self):
        timestamp = datetime.now().strftime("%B-%d_%H-%M")
        self.run_name = f"{timestamp}_{self.model_name}_lr-{self.learning_rate}_bs-{self.batch_size}_epochs-{self.cl_epochs}_seed-{self.seed}"
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

    print("Tokenizing labeled data for classification...")
    tok_labeled = ds.map(
        tokenize_classification,
        batched=True,
        remove_columns=[col for col in raw["train"].column_names if col != "label"],
    )
    tok_labeled = tok_labeled.remove_columns(["label"])


    # 2. MLM tokenization with validation split
    def tokenize_mlm(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    print("Tokenizing unlabeled data for MLM...")
    if cfg.mlm_validation_split > 0:
        # Use cfg.seed for the split
        mlm_splits = raw["train"].train_test_split(
            test_size=cfg.mlm_validation_split,
            seed=cfg.seed,
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    print(f"Successfully loaded model. Type: {type(model)}")
    model.resize_token_embeddings(len(tokenizer))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

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
    plt.figure(figsize=(max(6, len(labels)), max(6, len(labels))))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

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
        mlm_probability=cfg.mlm_probability
    )
    mlm_training_args = TrainingArguments(
        output_dir=cfg.mlm_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.mlm_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        save_steps=10000,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=cfg.mlm_learning_rate,
        weight_decay=cfg.weight_decay,
        report_to="wandb",
        load_best_model_at_end=True if "validation" in tok_unlabeled_data else False,
        eval_strategy="epoch" if "validation" in tok_unlabeled_data else "no",
        save_strategy="epoch" if "validation" in tok_unlabeled_data else "steps",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )
    mlm_trainer_kwargs = {
        "model": mlm_model,
        "args": mlm_training_args,
        "train_dataset": tok_unlabeled_data["train"],
        "data_collator": mlm_data_collator,
    }
    
    if "validation" in tok_unlabeled_data:
        mlm_trainer_kwargs["eval_dataset"] = tok_unlabeled_data["validation"]
    
    mlm_trainer = Trainer(**mlm_trainer_kwargs)
    
    print("Starting MLM training")
    mlm_trainer.train()
    
    if "validation" in tok_unlabeled_data:
        try:
            print("Evaluating MLM model for perplexity...")
            metrics = mlm_trainer.evaluate(eval_dataset=tok_unlabeled_data["validation"])
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                final_perplexity = np.exp(eval_loss)
                wandb.log({"final_mlm_perplexity": final_perplexity, "mlm_eval_loss": eval_loss})
                print(f"Final MLM perplexity: {final_perplexity:.4f} (from eval_loss: {eval_loss:.4f})")
            else:
                print("Could not find 'eval_loss' in MLM trainer evaluation metrics.")
        except Exception as e:
            print(f"Could not compute perplexity from trainer.evaluate(): {e}")
    
    print(f"Saving domain-adapted MLM model to {cfg.mlm_output_dir}")
    mlm_trainer.save_model(cfg.mlm_output_dir)
    tokenizer.save_pretrained(cfg.mlm_output_dir)
    print(f"Domain adaptation complete. Model saved to {cfg.mlm_output_dir}")
    
    return cfg.mlm_output_dir


# ------------------
# Main training pipeline
# ------------------
def main():
    cfg = Config()
    
    # Set all seeds for reproducibility
    set_seed(cfg.seed)
    print(f"Global seed set to {cfg.seed}")

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
    id2label = {v: k for k,v in label2id.items()}


    # --- Domain Adaptation (Masked Language Model) ---
    if cfg.enable_domain_adapt:
        adapted_model_path = perform_domain_adaptation(cfg, tok_unlabeled_data, tokenizer)
    else:
        adapted_model_path = cfg.model_name
        print("Skipping domain adaptation")

    # --- Text Classification ---
    print("Building classification model...")
    cl_model, compute_metrics_fn = build_model_and_metrics(adapted_model_path, num_labels, tokenizer)
    cl_model.config.id2label = id2label # For better inference later if needed
    cl_model.config.label2id = label2id
    
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
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )
    cl_trainer = Trainer(
        model=cl_model,
        args=cl_training_args,
        train_dataset=tok_labeled_data["train"],
        eval_dataset=tok_labeled_data["validation"],
        data_collator=cl_data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )
    print("Starting classification training")
    cl_trainer.train()

    # Evaluate on test set
    print("Evaluating on test set")
    pred_output = cl_trainer.predict(tok_labeled_data["test"])
    
    wandb.log({f"test_{k}": v for k, v in pred_output.metrics.items()})

    y_true_test = pred_output.label_ids
    y_pred_test_logits = pred_output.predictions
    y_pred_test_labels = np.argmax(y_pred_test_logits, axis=-1)
    
    cm = confusion_matrix(y_true_test, y_pred_test_labels)

    # Save model & results
    print("Saving model and results")
    cl_trainer.save_model() # This also saves tokenizer if passed to Trainer
    # tokenizer.save_pretrained(cfg.output_dir) # Redundant if tokenizer passed to Trainer

    results = {
        "test_metrics": pred_output.metrics,
        "confusion_matrix": cm.tolist(),
        "label2id": label2id,
        "id2label": id2label,
        "domain_adapted": cfg.enable_domain_adapt,
        "config": vars(cfg)
    }
    os.makedirs(cfg.output_dir, exist_ok=True) # Ensure dir exists
    with open(os.path.join(cfg.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    cm_path = os.path.join(cfg.output_dir, "confusion_matrix_test.png")
    plot_and_save_confusion_matrix(cm, list(label2id.keys()), cm_path)
    wandb.log({"test_confusion_matrix": wandb.Image(cm_path)})

    raw_test_texts = load_dataset(cfg.dataset_name, split="test")["text"]


    misclassification_table = wandb.Table(columns=["text", "true_label", "predicted_label"])
    for i in range(len(raw_test_texts)):
        if i < len(y_true_test):
            true_label_id = y_true_test[i]
            pred_label_id = y_pred_test_labels[i]
            if true_label_id != pred_label_id:
                misclassification_table.add_data(
                    raw_test_texts[i], 
                    id2label[true_label_id], 
                    id2label[pred_label_id],
                )
    wandb.log({"test_misclassifications": misclassification_table})

    y_pred_test_probas = F.softmax(torch.tensor(y_pred_test_logits), dim=-1).numpy()
    try:
        wandb.log({
            "test_roc_curve": wandb.plot.roc_curve(y_true_test, y_pred_test_probas, labels=list(label2id.keys())),
            "test_pr_curve": wandb.plot.pr_curve(y_true_test, y_pred_test_probas, labels=list(label2id.keys())),
        })
    except Exception as e:
        print(f"Could not log multiclass ROC/PR curves: {e}")
        print("This might happen if a class has no true samples in y_true_test or wandb version issue.")


    wandb.finish()
    print("Training and evaluation complete. Results saved and logged to W&B.")


if __name__ == "__main__":
    main()