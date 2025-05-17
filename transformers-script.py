import os
import logging
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

#!pip install --upgrade --no-deps evaluate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import evaluate
import wandb
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from kaggle_secrets import UserSecretsClient

# ------------------
# Configuration
# ------------------
@dataclass
class Config:
    model_name: str = "roberta-base"
    dataset_name: str = "Intel/polite-guard"
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    wandb_project: str = "polite-guard-classification"
    run_name: str = f"{datetime.now().strftime('%B-%d_%H-%M')}_{model_name}_lr-{learning_rate}_bs-{batch_size}_epochs-{num_epochs}"
    output_dir: str = f"./training_output/{model_name}/{run_name}"


# ------------------
# Setup logging & secrets
# ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_secrets():
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    hf_token = secrets.get_secret("huggingface")
    os.environ["HUGGINGFACE_TOKEN"] = hf_token


# ------------------
# Data Loading & Preprocessing
# ------------------

def load_and_preprocess(cfg: Config):
    logger.info("Loading dataset %s", cfg.dataset_name)
    raw = load_dataset(cfg.dataset_name)
    ds = DatasetDict({
        "train": raw["train"],
        "validation": raw["validation"],
        "test": raw["test"],
    })

    label_list = sorted(raw["train"].unique("label"))
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess(batch):
        toks = tokenizer(batch["text"], truncation=True, padding=True)
        toks["labels"] = [label2id[l] for l in batch["label"]]
        return toks

    logger.info("Tokenizing dataset")
    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    return tokenized, tokenizer, label2id


# ------------------
# Model & Metrics
# ------------------

def build_model_and_metrics(cfg: Config, num_labels: int):
    logger.info("Loading model %s", cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )

    metric = evaluate.load("f1")

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
# Main training pipeline
# ------------------

def main():
    cfg = Config()
    setup_secrets()

    # Initialize W&B
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=vars(cfg),
        save_code=True,
    )

    # Load & preprocess
    tokenized, tokenizer, label2id = load_and_preprocess(cfg)
    num_labels = len(label2id)

    # Model & metrics
    model, compute_metrics = build_model_and_metrics(cfg, num_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training")
    trainer.train()

    # Evaluate
    logger.info("Evaluating on test set")
    pred_output = trainer.predict(tokenized["test"])
    cm = confusion_matrix(pred_output.label_ids, np.argmax(pred_output.predictions, axis=-1))

    # Save model & results
    logger.info("Saving model and results")
    trainer.save_model()
    os.makedirs(cfg.output_dir, exist_ok=True)

    results = {
        "metrics": pred_output.metrics,
        "confusion_matrix": cm.tolist(),
        "label2id": label2id,
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

    last_eval = [h for h in trainer.state.log_history if h.get("eval_loss")][-1]
    eval_table = wandb.Table(columns=["metric", "value"])
    for k, v in last_eval.items():
        if k.startswith("eval_"):
            eval_table.add_data(k, v)
    wandb.log({"eval_metrics_table": eval_table})

    wandb.log({
        "roc": wandb.plot.roc_curve(pred_output.label_ids,pred_output.predictions,labels=list(label2id.keys())),
        "pr": wandb.plot.pr_curve(pred_output.label_ids,pred_output.predictions,labels=list(label2id.keys())),
    })

    wandb.finish()


if __name__ == "__main__":
    main()
