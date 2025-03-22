import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import re
import gc
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union, Set


# Download NLTK resources if not already available
def download_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet')
    ]
    
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


class TextPreprocessor:
    """
    Handles text preprocessing for NLP tasks including tokenization,
    lemmatization, and POS tagging.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        
    def get_wordnet_pos(self, nltk_tag: str) -> str:
        """
        Convert NLTK POS tags to WordNet POS tags.
        
        Args:
            nltk_tag: NLTK POS tag
            
        Returns:
            Corresponding WordNet POS tag
        """
        tag_to_wordnet = {
            'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.ADV
        }
        return tag_to_wordnet.get(nltk_tag[0], wordnet.NOUN)
    
    def preprocess_text(self, text: str, include_sw: bool, include_digits: bool) -> str:
        """
        Preprocess text data for NLP tasks.
        
        Args:
            text: Raw text to preprocess
            include_sw: Whether to include stopwords
            include_digits: Whether to include digits
            
        Returns:
            Preprocessed text
        """
        # Handle None or empty strings
        if not text or pd.isna(text):
            return ""
            
        # Remove non-alphabetic characters (and optionally digits)
        if not include_digits:
            text = re.sub('[^a-zA-Z]', ' ', text)
        else:
            text = re.sub('[^a-zA-Z0-9]', ' ', text)
            
        text = text.lower()

        # Tokenize
        words = word_tokenize(text)

        # POS tagging
        tagged_words = pos_tag(words)

        # Lemmatize with POS tags and handle stopwords based on the include_sw flag
        lemmatized_words = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)) 
            for word, pos_tag in tagged_words 
            if word not in self.stopwords or include_sw
        ]
        
        return ' '.join(lemmatized_words)


class DataLoader:
    """
    Handles loading and processing of data files in batches to manage memory efficiently.
    """
    def __init__(self, file_paths: Dict[str, str], preprocessor: TextPreprocessor):
        self.file_paths = file_paths
        self.preprocessor = preprocessor
        
    def process_in_batches(self, 
                          paths: List[str], 
                          feature: str, 
                          target: str, 
                          include_sw: bool, 
                          include_digits: bool, 
                          batch_size: int = 1000) -> Tuple[List[str], List[int]]:
        """
        Process files in batches to avoid memory issues.
        
        Args:
            paths: List of file paths to process
            feature: Column name containing text data
            target: Column name containing labels
            include_sw: Whether to include stopwords in preprocessing
            include_digits: Whether to include digits in preprocessing
            batch_size: Number of records to process at once
            
        Returns:
            Tuple of processed texts and corresponding labels
        """
        processed_texts = []
        labels = []
        
        for path in paths:
            print(f"Processing file: {path}")
            # Process file in chunks
            for chunk in pd.read_csv(path, chunksize=batch_size):
                chunk_texts = chunk[feature].fillna('').apply(
                    lambda x: self.preprocessor.preprocess_text(x, include_sw, include_digits)
                )
                processed_texts.extend(chunk_texts.tolist())
                labels.extend(chunk[target].tolist())
                
                # Free memory
                del chunk_texts
                gc.collect()
        
        return processed_texts, labels


class FeatureExtractor:
    """
    Handles feature extraction from text data using various vectorization methods.
    """
    def __init__(self, extractor_type: str, params: Dict[str, Any]):
        """
        Initialize the feature extractor.
        
        Args:
            extractor_type: Type of extractor (BoW, TF-IDF, etc.)
            params: Parameters for the extractor
        """
        self.extractor_type = extractor_type
        
        if extractor_type.startswith("BoW"):
            self.vectorizer = CountVectorizer(**params)
        elif extractor_type == "TF-IDF":
            self.vectorizer = TfidfVectorizer(**params)
        else:
            raise ValueError(f"Unsupported extractor type: {extractor_type}")
            
    def extract_features(self, texts: List[str], train: bool = False):
        """
        Extract features while keeping matrices sparse.
        
        Args:
            texts: List of processed text documents
            train: Whether this is training data (fit_transform) or not (transform)
            
        Returns:
            Sparse feature matrix
        """
        if train:
            return self.vectorizer.fit_transform(texts)
        else:
            return self.vectorizer.transform(texts)


class DimensionalityReducer:
    """
    Handles dimensionality reduction for feature matrices.
    """
    def __init__(self, method: str = "SVD", n_components: int = 200):
        self.method = method
        self.n_components = n_components
        
        if method == "SVD":
            self.reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
            
    def reduce_dimensions(self, X, train: bool = False):
        if train:
            return self.reducer.fit_transform(X)
        else:
            return self.reducer.transform(X)


class FeatureScaler:
    """
    Handles scaling of feature matrices.
    """
    def __init__(self, method: str = "standard", with_mean: bool = True, requires_non_negative: bool = False):
        self.method = method
        self.with_mean = with_mean
        
        if requires_non_negative:
            self.scaler = MinMaxScaler()
        elif method == "standard":
            self.scaler = StandardScaler(with_mean=with_mean)
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
            
    def scale_features(self, X, train: bool = False):
        if train:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)


class ModelTrainer:
    """
    Handles model training and hyperparameter tuning.
    """
    def __init__(self, model_type: str, hyperparameters: Dict[str, List[Any]] = None):
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        
        if model_type == "SVM":
            self.model = SVC(random_state=42)
        elif model_type == "NB":
            self.model = MultinomialNB()
        elif model_type == "LR":
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def train_model(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        
        return self.model, training_time
        
    def evaluate_model(self, X_val, y_val):
        """
        Evaluate model on validation data, returning accuracy score.
        """
        y_pred = self.model.predict(X_val)
        return accuracy_score(y_val, y_pred)
        
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Tune hyperparameters using grid search.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Validation feature matrix
            y_val: Validation labels
            
        Returns:
            Best parameters, validation score, and training time
        """
        best_score = 0
        best_params = None
        best_training_time = 0
        
        # If no hyperparameters to tune or model is SVM, just train once with default parameters
        if not self.hyperparameters or self.model_type == "SVM":
            model, training_time = self.train_model(X_train, y_train)
            score = self.evaluate_model(X_val, y_val)
            return {} if self.model_type == "SVM" else self.model.get_params(), score, training_time
        
        # Otherwise, tune hyperparameters
        for params in ParameterGrid(self.hyperparameters):
            self.model.set_params(**params)
            
            model, training_time = self.train_model(X_train, y_train)
            score = self.evaluate_model(X_val, y_val)
            
            print(f"Score: {score:.4f}, Time: {training_time:.2f}s for {params}")
            if score > best_score:
                best_score = score
                best_params = params
                best_training_time = training_time
        
        return best_params, best_score, best_training_time


class ExperimentTracker:
    """
    Handles tracking and saving of experiment results.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.results = []
        self.completed_configs = set()
        
        # Load existing results if file exists
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                self.results = existing_df.to_dict('records')
                
                # Create a set of completed configurations
                for result in self.results:
                    config_key = (
                        result['include_digits'],
                        result['feature_extraction'],
                        result['dim'],
                        result['model']
                    )
                    self.completed_configs.add(config_key)
                
                print(f"Loaded {len(self.results)} existing results from {filename}")
            except Exception as e:
                print(f"Error loading existing results: {e}")
    
    def is_completed(self, config_key: Tuple) -> bool:
        return config_key in self.completed_configs
        
    def add_result(self, result: Dict[str, Any]):
        self.results.append(result)
        
        # Add to completed configs
        config_key = (
            result['include_digits'],
            result['feature_extraction'],
            result['dim'],
            result['model']
        )
        self.completed_configs.add(config_key)
        
        # Save results incrementally
        self.save_results()
        
    def save_results(self):
        pd.DataFrame(self.results).to_csv(self.filename, index=False)
        print(f"Results saved to {self.filename}")
        
    def get_best_config(self):
        results_df = pd.DataFrame(self.results)
        best_config = results_df.loc[results_df['val_score'].idxmax()]
        return best_config


class ExperimentRunner:
    """
    Main class that orchestrates the entire NLP pipeline experiment.
    """
    def __init__(self, file_paths: Dict[str, str], results_filename: str):
        download_nltk_resources()
        
        self.file_paths = file_paths
        self.preprocessor = TextPreprocessor()
        self.data_loader = DataLoader(file_paths, self.preprocessor)
        self.tracker = ExperimentTracker(results_filename)
        
    def setup_experiment(self, 
                         include_digits_options: Tuple[bool, ...],
                         feature_extraction_configs: Tuple[Dict[str, Any], ...],
                         dimension_options: Tuple[int, ...],
                         model_configs: Tuple[Dict[str, Any], ...]):
        """
        Set up experiment configurations.
        
        Args:
            include_digits_options: Options for including digits
            feature_extraction_configs: Configurations for feature extraction
            dimension_options: Options for dimensionality reduction
            model_configs: Configurations for models
        """
        self.include_digits_options = include_digits_options
        self.feature_extraction_configs = feature_extraction_configs
        self.dimension_options = dimension_options
        self.model_configs = model_configs
        
    def run_experiment(self, feature: str = 'text', target: str = 'label'):
        """
        Run the experiment with all configurations.
        
        Args:
            feature: Column name containing text data
            target: Column name containing labels
        """
        train_files = [self.file_paths['train_cot'], self.file_paths['train_few_shot']]
        val_files = [self.file_paths['val_cot'], self.file_paths['val_few_shot']]
        
        # Loop through all experiment combinations
        for include_digit in self.include_digits_options:
            print(f"\n{'='*50}")
            print(f"Processing with include_digit={include_digit}")
            
            # Process data once for each include_digit setting
            processed_data = {}
            
            for feature_extraction in self.feature_extraction_configs:
                print(f"\n{'-'*50}")
                print(f"Feature extraction: {feature_extraction['name']}")
                
                # Get include_sw flag from feature_extraction
                include_sw = feature_extraction['include_sw']
                
                for dim in self.dimension_options:
                    # Determine dim name
                    dim_name = "full" if dim == -1 else f"SVD_{dim}"
                    
                    for model_config in self.model_configs:
                        # Check if this configuration has already been completed
                        config_key = (
                            include_digit,
                            feature_extraction['name'],
                            dim_name,
                            model_config['name']
                        )
                        
                        if self.tracker.is_completed(config_key):
                            print(f"Skipping already completed configuration: {config_key}")
                            continue
                        
                        print(f"\n{'-'*30}")
                        print(f"Model: {model_config['name']}")
                        
                        try:
                            # Run this specific experiment configuration
                            self._run_single_experiment(
                                include_digit, include_sw, feature_extraction, dim, dim_name,
                                model_config, processed_data, train_files, val_files, 
                                feature, target
                            )
                        except Exception as e:
                            print(f"Error in experiment: {e}")
                            continue
        
        # Print summary of best configurations
        best_config = self.tracker.get_best_config()
        print("\nBest configuration by validation score:")
        print(best_config)
    
    def _run_single_experiment(self,
                              include_digit: bool,
                              include_sw: bool,
                              feature_extraction: Dict[str, Any],
                              dim: int,
                              dim_name: str,
                              model_config: Dict[str, Any],
                              processed_data: Dict[str, Any],
                              train_files: List[str],
                              val_files: List[str],
                              feature: str,
                              target: str):
        """
        Run a single experiment configuration.
        
        Args:
            include_digit: Whether to include digits
            include_sw: Whether to include stopwords
            feature_extraction: Configuration for feature extraction
            dim: Dimension for dimensionality reduction
            dim_name: Name for dimension
            model_config: Configuration for model
            processed_data: Cached processed data
            train_files: Training files
            val_files: Validation files
            feature: Column name containing text data
            target: Column name containing labels
        """
        # Process data if not already processed for this include_digit setting
        data_key = f"data_{include_digit}_{include_sw}"
        if data_key not in processed_data:
            print(f">> Processing data with include_digits={include_digit}, include_sw={include_sw}")
            
            train_texts, train_labels = self.data_loader.process_in_batches(
                train_files, feature, target, include_sw=include_sw, include_digits=include_digit
            )
            
            val_texts, val_labels = self.data_loader.process_in_batches(
                val_files, feature, target, include_sw=include_sw, include_digits=include_digit
            )
            
            processed_data[data_key] = {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'val_texts': val_texts,
                'val_labels': val_labels
            }
        else:
            cached_data = processed_data[data_key]
            train_texts = cached_data['train_texts']
            train_labels = cached_data['train_labels']
            val_texts = cached_data['val_texts']
            val_labels = cached_data['val_labels']
        
        # Create feature extractor
        feature_extractor = FeatureExtractor(
            feature_extraction['name'],
            feature_extraction['params']
        )
        
        # Extract features
        print(">> Extracting features for training data")
        X_train = feature_extractor.extract_features(train_texts, train=True)
        print(f"Training data shape: {X_train.shape}")
        
        print(">> Extracting features for validation data")
        X_val = feature_extractor.extract_features(val_texts)
        print(f"Validation data shape: {X_val.shape}")
        
        # Reduce dimensions if needed
        if dim == -1:
            X_train_red = X_train
            X_val_red = X_val
        else:
            print(f">> Applying dimensionality reduction with {dim} components")
            
            try:
                # Use TruncatedSVD for sparse matrices
                dim_reducer = DimensionalityReducer(method="SVD", n_components=dim)
                
                # Process training data
                X_train_red = dim_reducer.reduce_dimensions(X_train, train=True)
                
                # Process validation data
                X_val_red = dim_reducer.reduce_dimensions(X_val)
                
                # Free memory
                gc.collect()
            except MemoryError:
                print("WARNING: Memory error during dimensionality reduction. Skipping this configuration.")
                return
        
        # Apply scaling based on model requirements
        if model_config['requires_non_negative'] and dim > 0:
            # For MultinomialNB after dimensionality reduction, ensure non-negative values
            print(">> Ensuring non-negative values for MultinomialNB")
            scaler = FeatureScaler(method="minmax", requires_non_negative=True)
            X_train_scaled = scaler.scale_features(X_train_red, train=True)
            X_val_scaled = scaler.scale_features(X_val_red)
        elif model_config['scale']:
            print(">> Scaling features")
            
            # Use appropriate scaler based on data type
            if isinstance(X_train_red, np.ndarray):
                # Dense matrix (after dimensionality reduction)
                scaler = FeatureScaler(method="standard", with_mean=True)
                X_train_scaled = scaler.scale_features(X_train_red, train=True)
                X_val_scaled = scaler.scale_features(X_val_red)
            else:
                # Sparse matrix (before dimensionality reduction)
                scaler = FeatureScaler(method="standard", with_mean=False)
                X_train_scaled = scaler.scale_features(X_train_red, train=True)
                X_val_scaled = scaler.scale_features(X_val_red)
        else:
            X_train_scaled = X_train_red
            X_val_scaled = X_val_red
        
        # Initialize model trainer
        model_trainer = ModelTrainer(
            model_config['name'],
            model_config['hyperparameters']
        )
        
        # Train and evaluate model
        best_params, val_score, training_time = model_trainer.tune_hyperparameters(
            X_train_scaled, train_labels, X_val_scaled, val_labels
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
        self.tracker.add_result(result)


def main():
    # Define file paths
    file_paths = {
        'train_cot': 'data/train/train_cot.csv',
        'train_few_shot': 'data/train/train_few_shot.csv',
        'val_cot': 'data/validation/val_cot.csv',
        'val_few_shot': 'data/validation/val_few_shot.csv',
    }
    
    # Initialize experiment runner
    output_dir = 'results'
    output_file = 'validation.csv'
    output_path = os.path.join(output_dir, output_file)
    runner = ExperimentRunner(file_paths, output_path)
    
    # Define experiment configurations
    include_digits_options = (True, False)
    embedding_dim = 200
    
    # Feature extraction configurations
    feature_extraction_configs = (
        {
            "name": "BoW_1",
            "params": {"ngram_range": (1, 1), "max_features": 50000},
            "include_sw": False
        },
        {
            "name": "BoW_2",
            "params": {"ngram_range": (2, 2), "max_features": 50000},
            "include_sw": True
        },
        {
            "name": "TF-IDF",
            "params": {"max_features": 50000},
            "include_sw": False
        }
    )
    
    # Dimensionality reduction options
    dimension_options = (-1, embedding_dim)
    
    # Model configurations
    model_configs = (
        {
            "name": "SVM",
            "hyperparameters": {},
            "scale": True,
            "requires_non_negative": False
        },
        {
            "name": "NB",
            "hyperparameters": {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]},
            "scale": False,
            "requires_non_negative": True
        },
        {
            "name": "LR",
            "hyperparameters": {
                'C': [0.1, 1.0, 10.0], 
                'max_iter': [100, 200, 300], 
                'penalty': ['l1', 'l2'], 
                'solver': ['liblinear']
            },
            "scale": True,
            "requires_non_negative": False
        },
    )
    
    # Set up and run experiment
    runner.setup_experiment(include_digits_options, feature_extraction_configs, 
                           dimension_options, model_configs)
    runner.run_experiment()


if __name__ == '__main__':
    main()