import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import re
import gc
import time

# Download NLTK resources if needed
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

# File paths
file_paths = {
    'train_cot': 'data/train/train_cot.csv',
    'train_few_shot': 'data/train/train_few_shot.csv',
    'test_cot': 'data/test/test_cot.csv',
    'test_few_shot': 'data/test/test_few_shot.csv',
    'test_LMs': 'data/test/test_lms.csv'
}

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words('english'))

# Function to convert NLTK POS tags to WordNet POS tags
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
    # Default is noun
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

    # Lemmatize with POS tags
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

def test_model(model, X_train, y_train, X_test, y_test):
    """
    Trains a given model on the training data and evaluates it on the test data.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    return test_accuracy, training_time

def main():
    # Best configuration parameters
    include_digits = False
    feature_extractor = TfidfVectorizer(max_features=50000, min_df=5)
    include_sw = False
    pca_dim = 200
    
    feature = 'text'
    target = 'label'
    
    print("Loading and processing data...")
    
    # Process training data
    train_files = [file_paths['train_cot'], file_paths['train_few_shot']]
    train_texts, train_labels = process_in_batches(
        train_files, feature, target, include_sw=include_sw, include_digits=include_digits
    )
    
    # Process test data
    test_files = [file_paths['test_cot'], file_paths['test_few_shot'], file_paths['test_LMs']]
    test_texts, test_labels = process_in_batches(
        test_files, feature, target, include_sw=include_sw, include_digits=include_digits
    )
    
    print(f"Training data: {len(train_texts)} samples")
    print(f"Test data: {len(test_texts)} samples")
    
    # Extract features
    print("Extracting features...")
    X_train, y_train = extract_features_sparse(
        train_texts, train_labels, feature_extractor, train=True
    )
    X_test, y_test = extract_features_sparse(
        test_texts, test_labels, feature_extractor, train=False
    )
    
    # Apply dimensionality reduction
    print(f"Applying SVD with {pca_dim} components...")
    svd = TruncatedSVD(n_components=pca_dim, random_state=42)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)
    
    # Train and test SVM model
    print("Training and testing SVM model...")
    svm = SVC(random_state=42)
    test_accuracy, training_time = test_model(
        svm, X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    print(f"\nResults for the best configuration:")
    print(f"Model: SVC")
    print(f"Feature extraction: TF-IDF")
    print(f"Dimension reduction: SVD_{pca_dim}")
    print(f"Include digits: {include_digits}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    
    # Save results
    results = {
        'Model': 'SVC',
        'Feature Extraction': 'TF-IDF',
        'Dimension Reduction': f'SVD_{pca_dim}',
        'Include Digits': include_digits,
        'Test Accuracy': test_accuracy,
        'Training Time': training_time
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('test_results_1.csv', index=False)
    print(f"Results saved to test_results_1.csv")

if __name__ == '__main__':
    main()