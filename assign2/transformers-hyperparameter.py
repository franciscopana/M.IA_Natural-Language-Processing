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
    method: str = "grid" 
    metric: Dict[str, str] = field(default_factory=lambda: {"name": "eval_f1", "goal": "maximize"})
    parameters: Dict[str, Dict] = field(default_factory=lambda: {
        "learning_rate": {"values": [1e-6, 1e-5, 2e-5]},  # 3 values
        "batch_size": {"values": [64]},                    # 1 value
        "weight_decay": {"values": [0.0, 0.01, 0.1]},      # 3 values
        "num_train_epochs": {"values": [1]},               # 1 value
    })
    max_trials: int = 10  

# ------------------
# Base Configuration
# ------------------

@dataclass
class BaseConfig:
    model_name: str = "roberta-base"
    #model_name: str = "bert-base-uncased"
    dataset_name: str = "Intel/polite-guard"
    wandb_project: str = "polite-guard-tuning"

# ------------------
# Setup secrets
# ------------------

def setup_secrets():
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    os.environ["HUGGINGFACE_TOKEN"] = secrets.get_secret("huggingface")


# ------------------
# Data Loading & Preprocessing
# ------------------

def load_and_preprocess(cfg: BaseConfig, tokenizer_for_preprocessing: AutoTokenizer, label2id_map: Dict[str, int]):
    print(f"Loading dataset {cfg.dataset_name}")
    raw = load_dataset(cfg.dataset_name)
    ds = DatasetDict({
        "train": raw["train"],
        "validation": raw["validation"],
        "test": raw["test"],
    })

    def preprocess(batch):
        toks = tokenizer_for_preprocessing(batch["text"], truncation=True, padding=True)
        toks["labels"] = [label2id_map[l] for l in batch["label"]] 
        return toks

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    return tokenized

def get_dataset_info(dataset_name: str):
    raw = load_dataset(dataset_name, split="train") 
    label_list = sorted(raw.unique("label"))
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    return label_list, label2id

# ------------------
# Model Builder & Metrics
# ------------------

def build_trainer(
    cfg: BaseConfig,
    tokenized: DatasetDict,
    tokenizer: AutoTokenizer, 
    num_labels: int,
    learning_rate: float,
    weight_decay: float,
    num_train_epochs: int,
    batch_size: int
) -> Trainer:
    print(f"Building model {cfg.model_name} with lr={learning_rate}, wd={weight_decay}, epochs={num_train_epochs}, bs={batch_size}")
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
        output_dir= f"./sweep_output/{wandb.run.id}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir=f"./logs/{wandb.run.id if wandb.run else 'default_run'}",
        report_to="wandb",
        fp16=True,
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
    return trainer

# ------------------
# Sweep Orchestration
# ------------------

base_cfg = BaseConfig()
sweep_cfg_global = SweepConfig() 

def sweep():
    setup_secrets()

    sweep_id = wandb.sweep(
        sweep={ 
            "method": sweep_cfg_global.method,
            "metric": sweep_cfg_global.metric,
            "parameters": sweep_cfg_global.parameters,
        },
        project=base_cfg.wandb_project,
    )

    def sweep_train(): 
        run = wandb.init(project=base_cfg.wandb_project) 

        lr = wandb.config.learning_rate      
        bs = wandb.config.batch_size
        wd = wandb.config.weight_decay
        ne = wandb.config.num_train_epochs

        run_name = f"{base_cfg.model_name.replace('/', '-')}_lr{lr}_wd{wd}_ep{ne}" 
        wandb.run.name = run_name                                                  
        wandb.run.save() 

        print(f"Starting W&B run: {run_name} with ID: {wandb.run.id if wandb.run else 'N/A'}")
        print(f"Hyperparameters: lr={lr}, bs={bs}, wd={wd}, ne={ne}")

        _, label2id = get_dataset_info(base_cfg.dataset_name)
        num_labels = len(label2id)

        tokenizer = AutoTokenizer.from_pretrained(base_cfg.model_name)

        tokenized_datasets = load_and_preprocess(base_cfg, tokenizer, label2id)


        trainer = build_trainer(
            base_cfg,
            tokenized_datasets,
            tokenizer, 
            num_labels,
            learning_rate=lr,
            weight_decay=wd,
            num_train_epochs=ne,
            batch_size=bs,
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        wandb.log(eval_metrics) 
        
        print(f"Run {run_name} finished. Metrics: {eval_metrics}")
        wandb.finish()

    wandb.agent(sweep_id, function=sweep_train, count=sweep_cfg_global.max_trials)

if __name__ == "__main__":
    sweep()