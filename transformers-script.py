"""
This script is not supposed to be ran in a local machine.
It's a merged version of a Kaggle notebook.
"""


print(">> Importing libraries")
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
!pip install evaluate
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import os
from kaggle_secrets import UserSecretsClient
import json


print(">> Experiment Parameters")
model_name = "roberta-base"
num_epochs = 1
batch_size = 64
learning_rate = 2e-5
weight_decay = 0.01
output_dir = f"./training_output/{model_name}/{datetime.now().strftime('%B-%d_%H-%M-%S')}"


print(">> Setting up for kaggle")
user_secrets = UserSecretsClient()
wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key
secret_value_0 = user_secrets.get_secret("huggingface")
!mkdir -p ~/.huggingface
!echo -n $huggingface_token > ~/.huggingface/token


print(">> Loading dataset")
raw_datasets = load_dataset("Intel/polite-guard")
train_valid_test_dataset = DatasetDict({
    'train': raw_datasets['train'],
    'validation': raw_datasets['validation'],
    'test': raw_datasets['test']
})


print(">> Preprocessing dataset")
label_list = raw_datasets["train"].unique("label")
label_list.sort()
label2id = {lbl: i for i, lbl in enumerate(label_list)}
id2label = {i: lbl for lbl, i in label2id.items()}


print(">> Label mapping")
def my_preprocess_function(tokenizer):
    def apply(sample):
        toks = tokenizer(sample["text"], truncation=True, padding=True)
        labels = [label2id[l] for l in sample["label"]]
        toks["labels"] = labels
        return toks
    return apply


print(">> Model setup")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = train_valid_test_dataset.map(
    my_preprocess_function(tokenizer),
    batched=True,
    remove_columns=["text", "source", "reasoning", "label"],
)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)


print(">> Evaluation metric setup")
metric = evaluate.load("f1")
def compute_f1(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")
def compute_confusion_matrix(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    cm = confusion_matrix(eval_pred.label_ids, predictions)
    return cm


print(">> Training setup")
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_f1,
)


print(">> Starting training")
trainer.train()


print(">> Getting predictions in test set")
predictions = trainer.predict(test_dataset=tokenized_dataset["test"])
cm = compute_confusion_matrix(predictions)


print(">> Saving model")
trainer.save_model()


print(">> Saving results")
results = {
    "model_name": model_name,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "test_f1": predictions.metrics["test_f1"],
    "confusion_matrix": cm.tolist(),
    "label2id": label2id,
}
with open(f'{output_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=4)


print(">> Confusion matrix")
labels = list(label2id.keys())
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
plt.yticks(np.arange(len(labels)), labels)

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png')
plt.show()


print(">> Saving misclassifications")
misclassified = []
for i, pred in enumerate(predictions.predictions):
    true_id = predictions.label_ids[i]
    pred_id = np.argmax(pred)
    if true_id != pred_id:
        misclassified.append({
            "text": train_valid_test_dataset["test"][i]["text"],
            "true_label": id2label[true_id],    
            "predicted_label": id2label[pred_id]
        })
with open(f'{output_dir}/misclassified.json', 'w') as f:
    json.dump(misclassified, f, indent=4)


print(">> Finished!")