import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
import gc  
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import time


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


file_paths = {
    'train_cot': 'data/train/train_cot.csv',
    'train_few_shot': 'data/train/train_few_shot.csv',
    'val_cot': 'data/validation/val_cot.csv',
    'val_few_shot': 'data/validation/val_few_shot.csv',
}


lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words('english'))


def get_wordnet_pos(nltk_tag):
    """
    Convert NLTK POS tags to WordNet POS tags.
    """
    tag_to_wordnet = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_to_wordnet.get(nltk_tag[0], wordnet.NOUN)


def preprocess_text(review, include_sw, include_digits):
    """
    Preprocess text data for NLP tasks
    """
    # Remove non-alphabetic characters (and optionally digits)
    review = re.sub('[^a-zA-Z]', ' ', review) if not include_digits else re.sub('[^a-zA-Z0-9]', ' ', review)
    review = review.lower()

    # Tokenize
    words = word_tokenize(review)

    # POS (Part-of-speech) tagging
    tagged_words = pos_tag(words)

    # Lemmatize with POS tags and handle stopwords based on the include_sw flag
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) 
                        for word, pos_tag in tagged_words 
                        if word not in sw or include_sw]
    return ' '.join(lemmatized_words)


def process_in_batches(file_paths, feature, target, include_sw, include_digits, batch_size=1000):
    """
    Process files in batches to avoid memory issues
    """
    processed_texts = []
    labels = []
    
    for path in file_paths:
        # Process file in chunks
        for chunk in pd.read_csv(path, chunksize=batch_size):
            chunk_texts = chunk[feature].fillna('').apply(
                lambda x: preprocess_text(x, include_sw, include_digits)
            )
            processed_texts.extend(chunk_texts.tolist())
            labels.extend(chunk[target].tolist())
            
            # Free memory
            del chunk_texts
            gc.collect()
    
    return processed_texts, labels


def extract_features_sparse(texts, labels, vectorizer, train=False):
    """
    Extract features while keeping matrices sparse
    """
    if train:
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    
    # Keep the data in sparse format
    return X, labels


def train_model_sparse(model, X_train, y_train, X_val, y_val):
    """
    Train model using sparse matrices and measure training time
    """
    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    # Calculate validation accuracy
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy, training_time


def tune_hyperparameters_sparse(model, X_train, y_train, X_val, y_val, hyperparam_grid):
    """
    Tune hyperparameters using sparse matrices and measure training time
    """
    best_score = 0
    best_params = None
    best_training_time = 0
    
    for params in ParameterGrid(hyperparam_grid):
        model.set_params(**params)
        
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        print(f"Score: {score:.4f}, Time: {training_time:.2f}s for {params}")
        if score > best_score:
            best_score = score
            best_params = params
            best_training_time = training_time
    
    return best_params, best_score, best_training_time


def run_experiment(include_digits, feature_extractions, dims, models, filename):
    # Check if results file exists and load existing results
    existing_results = []
    completed_configs = set()
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            existing_results = existing_df.to_dict('records')
            
            # Create a set of completed configurations
            for result in existing_results:
                config_key = (
                    result['include_digits'],
                    result['feature_extraction'],
                    result['dim'],
                    result['model']
                )
                completed_configs.add(config_key)
            
            print(f"Loaded {len(existing_results)} existing results from {filename}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Initialize results list with existing results
    results = existing_results
    
    feature = 'text'
    target = 'label'
    
    # Define file paths
    train_files = [file_paths['train_cot'], file_paths['train_few_shot']]
    val_files = [file_paths['val_cot'], file_paths['val_few_shot']]
    
    # Loop through all experiment combinations
    for include_digit in include_digits:
        print(f"\n{'='*50}")
        print(f"Processing with include_digit={include_digit}")
        
        # Process data once for each include_digit setting
        processed_data = {}
        
        for feature_extraction in feature_extractions:
            print(f"\n{'-'*50}")
            print(f"Feature extraction: {feature_extraction['name']}")
            
            # Get include_sw flag from feature_extraction
            include_sw = feature_extraction['include_sw']
            
            for dim in dims:
                # Determine dim name
                dim_name = "full" if dim == -1 else f"SVD_{dim}"
                
                for model_config in models:
                    # Check if this configuration has already been completed
                    config_key = (
                        include_digit,
                        feature_extraction['name'],
                        dim_name,
                        model_config['name']
                    )
                    
                    if config_key in completed_configs:
                        print(f"Skipping already completed configuration: {config_key}")
                        continue
                    
                    print(f"\n{'-'*30}")
                    print(f"Model: {model_config['name']}")
                    
                    # Process data if not already processed for this include_digit setting
                    if 'train_texts' not in processed_data:
                        print(">> Processing training data")
                        train_texts, train_labels = process_in_batches(
                            train_files, feature, target, include_sw=include_sw, include_digits=include_digit
                        )
                        
                        print(">> Processing validation data")
                        val_texts, val_labels = process_in_batches(
                            val_files, feature, target, include_sw=include_sw, include_digits=include_digit
                        )
                        
                        processed_data['train_texts'] = train_texts
                        processed_data['train_labels'] = train_labels
                        processed_data['val_texts'] = val_texts
                        processed_data['val_labels'] = val_labels
                    else:
                        train_texts = processed_data['train_texts']
                        train_labels = processed_data['train_labels']
                        val_texts = processed_data['val_texts']
                        val_labels = processed_data['val_labels']
                    
                    # Extract features
                    print(">> Extracting features for training data")
                    X_train, y_train = extract_features_sparse(
                        train_texts, train_labels, 
                        feature_extraction['extractor'], 
                        train=True
                    )
                    print(f"Training data shape: {X_train.shape}")
                    
                    print(">> Extracting features for validation data")
                    X_val, y_val = extract_features_sparse(
                        val_texts, val_labels, 
                        feature_extraction['extractor'], 
                        train=False
                    )
                    print(f"Validation data shape: {X_val.shape}")
                    
                    # Skip dim reduction if dimension is -1
                    if dim == -1:
                        X_train_red = X_train
                        X_val_red = X_val
                    else:
                        print(f">> Applying dimensionality reduction with {dim} components")
                        
                        try:
                            # Use TruncatedSVD for sparse matrices 
                            svd = TruncatedSVD(n_components=dim, random_state=42)
                            
                            # Process training data
                            X_train_reduced = svd.fit_transform(X_train)
                            
                            # Process validation data
                            X_val_reduced = svd.transform(X_val)
                            
                            X_train_red = X_train_reduced
                            X_val_red = X_val_reduced
                            
                            # Free memory
                            gc.collect()
                        except MemoryError:
                            print("WARNING: Memory error during dimensionality reduction. Skipping this configuration.")
                            continue
                    
                    # Apply scaling based on model requirements
                    if model_config['requires_non_negative'] and dim > 0:
                        # For MultinomialNB after dimensionality reduction, ensure non-negative values
                        print(">> Ensuring non-negative values for MultinomialNB")
                        scaler = MinMaxScaler()
                        X_train_scaled = scaler.fit_transform(X_train_red)
                        X_val_scaled = scaler.transform(X_val_red)
                    elif model_config['scale']:
                        print(">> Scaling features")
                        
                        # Use appropriate scaler based on data type
                        if isinstance(X_train_red, np.ndarray):
                            # Dense matrix (after dimensionality reduction)
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train_red)
                            X_val_scaled = scaler.transform(X_val_red)
                        else:
                            # Sparse matrix (before dimensionality reduction)
                            scaler = StandardScaler(with_mean=False)
                            X_train_scaled = scaler.fit_transform(X_train_red)
                            X_val_scaled = scaler.transform(X_val_red)
                    else:
                        X_train_scaled = X_train_red
                        X_val_scaled = X_val_red
                    
                    # Initialize model with appropriate parameters
                    model_class = model_config['model']
                    if model_config['name'] == 'LR':
                        model = model_class(random_state=42)
                    else:
                        model = model_class()
                    
                    # For SVM, use default parameters and only measure training time
                    if model_config['name'] == 'SVM':
                        val_score, training_time = train_model_sparse(
                            model, X_train_scaled, y_train, X_val_scaled, y_val
                        )
                        best_params = 'NaN'  # As requested for SVM
                        print(f"SVM default parameters - Validation score: {val_score:.4f}, Training time: {training_time:.2f} seconds")
                    else:
                        # For other models, tune hyperparameters and measure training time
                        print(">> Tuning hyperparameters")
                        best_params, val_score, training_time = tune_hyperparameters_sparse(
                            model, X_train_scaled, y_train, X_val_scaled, y_val, 
                            model_config['hyperparameters']
                        )
                        print(f"Best parameters: {best_params}, validation score: {val_score:.4f}, training time: {training_time:.2f} seconds")
                    
                    # Store results
                    result = {
                        'include_digits': include_digit,
                        'feature_extraction': feature_extraction['name'],
                        'dim': dim_name,
                        'model': model_config['name'],
                        'val_score': val_score,
                        'training_time': training_time,
                        'best_params': str(best_params)
                    }
                    results.append(result)
                    
                    # Add to completed configs
                    completed_configs.add(config_key)
                    
                    # Save results incrementally
                    pd.DataFrame(results).to_csv(filename, index=False)
                    print(f"Results saved to {filename}")
    
    # Find best configuration
    results_df = pd.DataFrame(results)
    best_config = results_df.loc[results_df['val_score'].idxmax()]
    print("\nBest configuration by validation score:")
    print(best_config)
    
    fastest_config = results_df.loc[results_df['training_time'].idxmin()]
    print("\nFastest configuration:")
    print(fastest_config)
    
    # Save final results
    results_df.to_csv(filename, index=False)
    print(f"Final results saved to {filename}")

    
if __name__ == '__main__':

    include_digits = (False, )
    embedding_dim = 200
    feature_extractions = (
        # {
        #     "extractor": CountVectorizer(ngram_range=(1,1), max_features=50000),
        #     "name": "BoW_1",
        #     "include_sw": False
        # },
        {
            "extractor": CountVectorizer(ngram_range=(2,2), max_features=50000),
            "name": "BoW_2",
            "include_sw": True
        },
        {
            "extractor": TfidfVectorizer(max_features=50000),
            "name": "TF-IDF",
            "include_sw": False
        }
    )
    dims = (-1, embedding_dim)
    models = (
        {
            "model": SVC,
            "name": "SVM",
            "hyperparameters": {},
            "scale": True,
            "requires_non_negative": False
        },
        {
            "model": MultinomialNB,
            "name": "NB",
            "hyperparameters": {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]},
            "scale": False,
            "requires_non_negative": True
        },
        {
            "model": LogisticRegression,
            "name": "LR",
            "hyperparameters": {'C': [0.1, 1.0, 10.0], 'max_iter': [100, 200, 300], 
                               'penalty':['l1', 'l2'], 'solver':['liblinear']},
            "scale": True,
            "requires_non_negative": False
        },
    )
    filename = 'val_results.csv'
    run_experiment(include_digits, feature_extractions, dims, models, filename)
    
