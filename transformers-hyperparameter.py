import os
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

!pip install --upgrade --no-deps evaluate
import numpy as np
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
from transformers.integrations import WandbCallback
from kaggle_secrets import UserSecretsClient

# ------------------
# Sweep Configuration
# ------------------
@dataclass
class SweepConfig:
    # Use random search and limit combinations to fit within Kaggle 9h GPU constraint
    method: str = "grid"  # grid, random, bayes
    metric: Dict[str, str] = field(default_factory=lambda: {"name": "eval_f1", "goal": "maximize"})
    parameters: Dict[str, Dict] = field(default_factory=lambda: {
        "learning_rate": {"values": [1e-6, 1e-5, 2e-5]},  # 3 values
        "batch_size": {"values": [64]},                    # 1 value
        "weight_decay": {"values": [0.0, 0.01, 0.1]},      # 3 values
        "num_train_epochs": {"values": [1]},               # 1 value
    })
    # Total possible combinations = 3 * 1 * 3 * 1 = 9
    max_trials: int = 10  # run only top 5 random combinations

# ------------------
# Base Configuration
# ------------------
@dataclass
class BaseConfig:
    #model_name: str = "roberta-base"
    model_name: str = "bert-base-uncased"
    dataset_name: str = "Intel/polite-guard"
    wandb_project: str = "polite-guard-tuning"

# ------------------
# Setup logging & secrets
# ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_secrets():
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    os.environ["HUGGINGFACE_TOKEN"] = secrets.get_secret("huggingface")

# ------------------
# Data Loading & Preprocessing
# ------------------

def load_and_preprocess(cfg: BaseConfig):
    logger.info(f"Loading dataset {cfg.dataset_name}")
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

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    return tokenized, label2id

# ------------------
# Model Builder & Metrics
# ------------------

def build_trainer(
    cfg: BaseConfig,
    tokenized: DatasetDict,
    num_labels: int,
    learning_rate: float,
    weight_decay: float,
    num_train_epochs: int,
    batch_size: int
) -> Trainer:
    logger.info(f"Building model {cfg.model_name} with lr={learning_rate}, wd={weight_decay}, epochs={num_train_epochs}, bs={batch_size}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir="./sweep_output",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir="./logs",
        report_to="wandb",
        fp16=True,  # Add this line
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[WandbCallback()],
    )
    return trainer

# ------------------
# Sweep Orchestration
# ------------------

def sweep():
    base_cfg = BaseConfig()
    sweep_cfg = SweepConfig()
    setup_secrets()

    # Initialize W&B sweep
    sweep_id = wandb.sweep(
        {
            "method": sweep_cfg.method,
            "metric": sweep_cfg.metric,
            "parameters": sweep_cfg.parameters,
        },
        project=base_cfg.wandb_project,
    )

    def sweep_train():
        # initialize W&B run for this trial
        wandb.init(project=base_cfg.wandb_project)
        # retrieve hyperparameters from the sweep
        lr = wandb.config.learning_rate
        bs = wandb.config.batch_size
        wd = wandb.config.weight_decay
        ne = wandb.config.num_train_epochs

        tokenized, label2id = load_and_preprocess(base_cfg)
        num_labels = len(label2id)

        trainer = build_trainer(
            base_cfg,
            tokenized,
            num_labels,
            learning_rate=lr,
            weight_decay=wd,
            num_train_epochs=ne,
            batch_size=bs,
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        wandb.log(eval_metrics)
        wandb.finish()

    wandb.agent(sweep_id, function=sweep_train, count=sweep_cfg.max_trials)

if __name__ == "__main__":
    sweep()
