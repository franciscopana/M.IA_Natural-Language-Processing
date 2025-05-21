import os
import time
import re
import argparse
import logging
import csv
from typing import Callable, Dict, List
from datetime import datetime
import re
import logging
from typing import List
import google.generativeai as genai
from datasets import load_dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
API_KEY_ENV = "GOOGLE_API_KEY"
MODEL_ENV = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-2.0-flash-lite"
SLEEP_INTERVAL = 1.5  
PROMPTS_DIR = "prompts"  
DEBUG_DIR = "debug_logs" 

def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def configure_api():
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(f"Please set the '{API_KEY_ENV}' environment variable.")
    genai.configure(api_key=api_key)
    logging.info("Gemini API configured.")

def load_prompt_template(name: str) -> str:
    """Read a prompt template from a .txt file.
       Files should contain a '{text}' placeholder for insertion."""
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_prompt_builders() -> Dict[str, Callable[[str], str]]:
    builders: Dict[str, Callable[[str], str]] = {}
    for key in ["zs", "fs", "fs_cot"]:
        template = load_prompt_template(key)
        builders_key = {
            "zs": "Zero-Shot",
            "fs": "Few-Shot",
            "fs_cot": "FS-CoT"
        }[key]
        builders[builders_key] = lambda text, tpl=template: tpl.format(text=text)
    return builders


def call_gemini(prompt: str) -> str:
    try:
        model_name = os.getenv(MODEL_ENV, DEFAULT_MODEL)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as err:
        logging.error("Gemini API error: %s", err)
        return ""

def parse_label(response: str, valid_labels: List[str]) -> str:
    text = response.lower()

    # 1) Look specifically for a “Label:” prefix
    m = re.search(r"Label:\s*([^\r\n]+)", response, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().lower()
        if candidate in valid_labels:
            return candidate

    # 2) Otherwise fall back to word-boundary matching, longest first
    for label in sorted(valid_labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label

    logging.warning("Could not parse label from response: '%s'", response)
    return "unknown"

def load_samples(dataset_name: str, split: str, num: int):
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    # 2) Convert to a list of example-dicts
    all_examples = list(ds)
    
    # 3) Just take the first `num` examples of whatever split
    samples = all_examples[:num]
    
    # 4) Fixed label set for Intel/polite-guard
    labels = ["polite", "somewhat polite", "neutral", "impolite"]
    
    logging.info("Loaded %d samples from split '%s'.", len(samples), split)
    return samples, labels


def evaluate(samples: List[dict], labels: List[str], prompts: Dict[str, Callable[[str], str]], dataset_name: str, split: str) -> Dict[str, float]:
    results = {name: {'y_true': [], 'y_pred': []} for name in prompts}

    # Create debug CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_directory_exists(DEBUG_DIR)
    csv_filename = os.path.join(DEBUG_DIR, f"debug_{dataset_name.replace('/', '_')}_{split}_{timestamp}.csv")

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample_id', 'text', 'true_label', 'prompt_type', 'prompt', 'response', 'predicted_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(samples, start=1):
            text = item['text']
            true = item.get('label', 'unknown') 
            if isinstance(true, int):
                true = labels[true] 
            elif true not in labels:
                logging.warning(f"True label '{true}' not in valid labels.")
                true = 'unknown'

            for name, build_prompt in prompts.items():
                prompt = build_prompt(text)
                resp = call_gemini(prompt)
                pred = parse_label(resp, labels)

                writer.writerow({
                    'sample_id': idx,
                    'text': text.replace('\n', ' '),
                    'true_label': true,
                    'prompt_type': name,
                    'prompt': prompt.replace('\n', ' '),
                    'response': resp.replace('\n', ' '),
                    'predicted_label': pred
                })

                results[name]['y_true'].append(true)
                results[name]['y_pred'].append(pred)
                csvfile.flush()

            time.sleep(SLEEP_INTERVAL)

    logging.info(f"Debug information saved to {csv_filename}")

    return {
        name: f1_score(data['y_true'], data['y_pred'], average='weighted', labels=labels, zero_division=0)
        for name, data in results.items()
    }

def plot_scores(scores: Dict[str, float], title: str):
    names, vals = zip(*scores.items())
    plt.figure(figsize=(8, 6))
    sns.set_palette("viridis")
    plt.bar(names, vals)
    plt.ylim(0, 1.05)
    plt.ylabel('Weighted F1')
    plt.title(title)
    for x, y in zip(names, vals):
        plt.text(x, y + 0.02, f"{y:.2f}", ha='center')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, f"f1_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate politeness classification via Gemini")
    parser.add_argument('--dataset', default='Intel/polite-guard')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--num', type=int, default=10)
    args = parser.parse_args()

    configure_api()
    ensure_directory_exists(DEBUG_DIR)
    
    prompts = get_prompt_builders()
    samples, labels = load_samples(args.dataset, args.split, args.num)
    scores = evaluate(samples, labels, prompts, args.dataset, args.split)
    logging.info("F1 scores: %s", scores)
    title = f"Prompt Comparison on {args.num} '{args.split}' Samples"
    plot_scores(scores, title)

if __name__ == '__main__':
    main()