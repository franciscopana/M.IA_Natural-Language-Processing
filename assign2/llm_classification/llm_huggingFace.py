!pip install -U bitsandbytes --quiet
!pip install -U accelerate --quiet
!pip install -U transformers --quiet
!pip install -U flash-attn --no-build-isolation --quiet

import os
import time
import re
import logging
import csv
from typing import Callable, Dict, List
from datetime import datetime
import torch
from threading import Lock
from huggingface_hub import login as hf_login
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
from datasets import load_dataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GemmaTokenizerFast
import transformers 

# --- Configuration & Logging ---
SECRETS = UserSecretsClient()
GOOGLE_API_KEY = SECRETS.get_secret("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = SECRETS.get_secret("Hugging_Face")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
SLEEP_INTERVAL = float(os.environ.get("SLEEP_INTERVAL", 1.5))
PROMPTS_DIR = os.environ.get("PROMPTS_DIR", "/kaggle/input/prompts-llm")
DEBUG_DIR = os.environ.get("DEBUG_DIR", "debug_logs")
MODELS_DIR = os.environ.get("MODELS_DIR", "cached_models")
gpu_lock = Lock()

# --- Models Configuration ---
class ModelConfig:
    def __init__(self, name: str, provider: str, model_id: str,
                 max_tokens: int = 2048, temperature: float = 0.0,
                 quantize: bool = False, device_map: str = "auto"):
        self.name = name
        self.provider = provider
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.quantize = quantize
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def __str__(self): return f"{self.name} ({self.provider})"

"""
"phi-3-mini-4k": ModelConfig(
    name="Phi-3 Mini 4K",
    provider="huggingface",
    model_id="microsoft/Phi-3-mini-4k-instruct",
    max_tokens=1024,
    quantize=True
),
"mistral-7b": ModelConfig(
    name="Mistral 7B",
    provider="huggingface",
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=1024,
    quantize=True
),
"gemma-7b": ModelConfig(
    name="Gemma 7B",
    provider="huggingface",
    model_id="google/gemma-7b-it",
    max_tokens=1024,
    quantize=True
),
"llama3-8b": ModelConfig(
            name="Llama 3 8B",
            provider="huggingface",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=1024,
            quantize=True
        ),
"qwen2-7b": ModelConfig(
            name="Qwen2 7B",
            provider="huggingface",
            model_id="Qwen/Qwen2-7B-Instruct",
            max_tokens=1024,
            quantize=True
        ),
"deepseek-7b": ModelConfig(
    name="DeepSeek 7B",
    provider="huggingface",
    model_id="TheBloke/deepseek-llm-7B-base-GPTQ",
    max_tokens=1024,
    quantize=True
),
"""
def get_available_models() -> Dict[str, ModelConfig]:
    return {
        "llama3-8b": ModelConfig(
            name="Llama 3 8B",
            provider="huggingface",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=1024,
            quantize=True
        ),
    }

def ensure_directory_exists(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def configure_api():
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Gemini API configured via Kaggle Secrets.")
    else:
        print("GOOGLE_API_KEY not found in Kaggle Secrets.")
    if HUGGINGFACE_API_KEY:
        hf_login(token=HUGGINGFACE_API_KEY)
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY
        print("Hugging Face API configured via Kaggle Secrets.")
    else:
        print("HUGGINGFACE_API_KEY not found in Kaggle Secrets.")
    ensure_directory_exists(MODELS_DIR)
    os.environ["TRANSFORMERS_CACHE"] = os.path.abspath(MODELS_DIR)
    print(f"Hugging Face cache set to {os.environ['TRANSFORMERS_CACHE']}")

def load_prompt_template(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()

def get_prompt_builders() -> Dict[str, Callable[[str], str]]:
    builders: Dict[str, Callable[[str], str]] = {}
    for key in ["zs", "fs", "fs_cot"]:
        tpl = load_prompt_template(key)
        builders_key = {"zs": "Zero-Shot", "fs": "Few-Shot", "fs_cot": "FS-CoT"}[key]
        builders[builders_key] = lambda text, t=tpl: t.format(text=text)
    return builders

def load_huggingface_model(config: ModelConfig):
    with gpu_lock:
        print(f"Loading Hugging Face model: {config.model_id}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        quant_cfg = transformers.BitsAndBytesConfig(
            load_in_4bit=config.quantize,
            bnb_4bit_compute_dtype=torch.float16
        ) if config.quantize else None

        tokenizer = None
        try:
            tokenizer = GemmaTokenizerFast.from_pretrained(
                config.model_id,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                force_download=True,
                local_files_only=False
            )
            print(f"Successfully loaded GemmaTokenizerFast for {config.model_id}")
        except Exception as e_gemma_fast:
            print(f"Error loading GemmaTokenizerFast: {e_gemma_fast}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_id,
                    trust_remote_code=True,
                    use_auth_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                    force_download=True,
                    local_files_only=False
                )
                print(f"Successfully loaded AutoTokenizer for {config.model_id}")
            except Exception as e_auto:
                print(f"Error loading AutoTokenizer: {e_auto}")
                raise

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            device_map=config.device_map,
            quantization_config=quant_cfg,
            trust_remote_code=True,
            use_auth_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
            force_download=True,
            local_files_only=False
        )
        config._model, config._tokenizer = model, tokenizer
        return model, tokenizer

def call_model(prompt: str, config: ModelConfig) -> str:
    try:
        if config.provider == "gemini": return call_gemini(prompt, config)
        if config.provider == "huggingface": return call_huggingface(prompt, config)
    except Exception as e:
        print(f"Error calling {config.name}: {e}")
    return ""

def call_gemini(prompt: str, config: ModelConfig) -> str:
    try:
        model = genai.GenerativeModel(config.model_id)
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": config.max_tokens, "temperature": config.temperature}
        )
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
    return ""

def call_huggingface(prompt: str, config: ModelConfig) -> str:
    try:
        if not config._model or not config._tokenizer:
            load_huggingface_model(config)
        gen = pipeline(
            "text-generation",
            model=config._model,
            tokenizer=config._tokenizer,
            max_length=len(config._tokenizer.encode(prompt)) + config.max_tokens,
            temperature=config.temperature,
            top_p=0.95,
            do_sample=config.temperature > 0,
            pad_token_id=config._tokenizer.eos_token_id
        )
        out = gen(prompt, max_new_tokens=config.max_tokens, num_return_sequences=1)[0]["generated_text"]
        resp = out[len(prompt):].strip() or out
        return resp
    except Exception as e:
        print(f"HF model error: {e}")
        if torch.cuda.is_available(): print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return ""

def parse_label(resp: str, labels: List[str]) -> str:
    low = resp.lower()
    m = re.search(r"Label:\s*([^\r\n]+)", resp, flags=re.IGNORECASE)
    if m and m.group(1).strip().lower() in labels:
        return m.group(1).strip().lower()
    for lab in sorted(labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(lab)}\b", low): return lab
    print(f"Could not parse label: {resp}")
    return "unknown"

def load_samples(ds: str, split: str, num: int):
    data = load_dataset(ds, split=split, trust_remote_code=True)
    exs = list(data)[:num]
    labs = ["polite", "somewhat polite", "neutral", "impolite"]
    print(f"Loaded {len(exs)} samples from {split}.")
    return exs, labs

def evaluate(
    samples: List[dict],
    labels: List[str],
    prompts: Dict[str, Callable[[str], str]],
    models: Dict[str, ModelConfig],
    ds: str,
    split: str
) -> Dict[str, Dict[str, float]]:
    results = {m: {p: {'y_true': [], 'y_pred': []} for p in prompts} for m in models}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_directory_exists(DEBUG_DIR)
    path = os.path.join(DEBUG_DIR, f"debug_{ds.replace('/', '_')}_{split}_{ts}.csv")
    with open(path, 'w', newline='', encoding='utf-8') as cf:
        w = csv.DictWriter(cf, fieldnames=[
            'sample_id','text','true_label','model','prompt_type','prompt','response','predicted_label'
        ])
        w.writeheader()
        for idx, itm in enumerate(samples, 1):
            txt = itm['text']
            true = itm.get('label', 'unknown')
            if isinstance(true, int): true = labels[true]
            if true not in labels: true = 'unknown'
            for mn, mc in models.items():
                for pn, bd in prompts.items():
                    prm = bd(txt)
                    rsp = call_model(prm, mc)
                    pred = parse_label(rsp, labels)
                    w.writerow({
                        'sample_id': idx,
                        'text': txt.replace('\n',' '),
                        'true_label': true,
                        'model': mn,
                        'prompt_type': pn,
                        'prompt': prm.replace('\n',' '),
                        'response': rsp.replace('\n',' '),
                        'predicted_label': pred
                    })
                    results[mn][pn]['y_true'].append(true)
                    results[mn][pn]['y_pred'].append(pred)
                    cf.flush()
                if mc.provider=='huggingface' and torch.cuda.is_available(): torch.cuda.empty_cache()
            time.sleep(SLEEP_INTERVAL)
    print(f"Saved debug CSV to {path}")
    f1s = {}
    for mn in models:
        f1s[mn] = {}
        for pn in prompts:
            y = results[mn][pn]
            f1s[mn][pn] = f1_score(y['y_true'], y['y_pred'], average='weighted', labels=labels, zero_division=0)
    
    # Save F1 scores to CSV
    safe_ds = ds.replace('/', '_')
    f1_csv_path = os.path.join('/kaggle/working', f"f1_scores_{safe_ds}_{split}_{ts}.csv")
    with open(f1_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'split', 'model', 'prompt_type', 'f1_score'])
        for mn in models:
            for pn in prompts:
                writer.writerow([ds, split, mn, pn, f1s[mn][pn]])
    print(f"Saved F1 scores to {f1_csv_path}")
    
    return f1s

def plot_scores(scores: Dict[str, Dict[str, float]], title: str):
    import numpy as np
    models = list(scores)
    pts = list(scores[models[0]])
    fig, ax = plt.subplots(figsize=(12,8))
    bw = 0.8/len(pts)
    idx = np.arange(len(models))
    for i, pt in enumerate(pts):
        vals = [scores[m][pt] for m in models]
        bars = ax.bar(idx + i*bw - (len(pts)-1)*bw/2, vals, bw, label=pt)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Models')
    ax.set_ylabel('Weighted F1 Score')
    ax.set_title(title)
    ax.set_xticks(idx)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Prompt Types')
    ax.set_ylim(0,1.05)
    plt.tight_layout()
    ensure_directory_exists(DEBUG_DIR)
    plt.savefig(os.path.join(DEBUG_DIR, f"f1_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.close()

def run_evaluation(
    dataset: str = os.environ.get('DATASET', 'Intel/polite-guard'),
    split: str = os.environ.get('SPLIT', 'validation'),
    num_samples: int = int(os.environ.get('NUM_SAMPLES', 10)),
    selected_models: List[str] = os.environ.get('MODELS', 'all').split(','),
    prompt_types: List[str] = os.environ.get('PROMPT_TYPES', 'Zero-Shot,Few-Shot,FS-CoT').split(','),
    plot: bool = True
):
    configure_api()
    ensure_directory_exists(DEBUG_DIR)
    ensure_directory_exists(PROMPTS_DIR)
    ensure_directory_exists(MODELS_DIR)
    models = get_available_models()
    if 'all' not in selected_models:
        models = {k: models[k] for k in selected_models if k in models}
    prompts = get_prompt_builders()
    prompts = {k: prompts[k] for k in prompt_types if k in prompts}
    samples, labels = load_samples(dataset, split, num_samples)
    print(f"Evaluating {len(models)} models on {num_samples} samples.")
    scores = evaluate(samples, labels, prompts, models, dataset, split)
    if plot:
        title = f"LLM Comparison: {num_samples} '{split}' samples"
        plot_scores(scores, title)
    print("Done.")
    return scores

# For Kaggle, call run_evaluation with plot=False to save results without plotting
df_scores = run_evaluation(plot=True)