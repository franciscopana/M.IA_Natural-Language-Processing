import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import gensim
import re
import gc
import os
import tensorflow as tf
from tensorboard.plugins import projector
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import Counter

def download_nltk_resources():
    """Downloads necessary NLTK resources if not found."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet')
    ]
    print("Checking NLTK resources...")
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"  Downloading '{resource_name}'...")
            nltk.download(resource_name, quiet=True)
    print("NLTK resource check complete.")

class TextPreprocessor:
    """Handles text preprocessing (tokenization, lemmatization, POS tagging)."""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def get_wordnet_pos(self, nltk_tag: str) -> str:
        tag_to_wordnet = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
        return tag_to_wordnet.get(nltk_tag[0], wordnet.NOUN)

    def preprocess_text(self, text: str, include_sw: bool, include_digits: bool) -> str:
        if not text or pd.isna(text): return ""
        pattern = '[^a-zA-Z0-9]' if include_digits else '[^a-zA-Z]'
        text = re.sub(pattern, ' ', text).lower()
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        lemmatized_words = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag))
            for word, pos_tag in tagged_words
            if include_sw or word not in self.stopwords
        ]
        return ' '.join(lemmatized_words)

class DataLoader:
    """Handles loading and processing of data files in batches."""
    def __init__(self, file_paths: Dict[str, str], preprocessor: TextPreprocessor):
        self.file_paths = file_paths
        self.preprocessor = preprocessor

    def process_in_batches(self, paths: List[str], feature: str, target: str,
                          include_sw: bool, include_digits: bool,
                          batch_size: int = 1000) -> Tuple[List[str], List[str]]:
        processed_texts = []
        labels = []
        for path in paths:
            print(f"  Processing file: {path}...")
            for chunk in pd.read_csv(path, chunksize=batch_size):
                if target not in chunk.columns:
                    print(f"Warning: Target column '{target}' not found in chunk from {path}. Skipping labels for this chunk.")
                    chunk_labels = ['Unknown'] * len(chunk) 
                else:
                   chunk_labels = chunk[target].astype(str).tolist()

                # Ensure feature column exists before using it
                if feature not in chunk.columns:
                     print(f"Warning: Feature column '{feature}' not found in chunk from {path}. Skipping texts for this chunk.")
                     chunk_texts = pd.Series([''] * len(chunk))
                else:
                    chunk_texts = chunk[feature].fillna('').apply(
                        lambda x: self.preprocessor.preprocess_text(x, include_sw, include_digits)
                    )

                processed_texts.extend(chunk_texts.tolist())
                labels.extend(chunk_labels)
                del chunk_texts, chunk; gc.collect() 
        return processed_texts, labels


# --- WordEmbeddingsModel Class ---
class WordEmbeddingsModel:
    """Handles Word2Vec training, loading, and saving for TensorBoard."""
    def __init__(self, vector_size: int, window: int, min_count: int, workers: int, sg: int, include_sw: bool, include_digits: bool):
        self.save_dir = "embeddings"
        os.makedirs(self.save_dir, exist_ok=True)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.include_sw = include_sw
        self.include_digits = include_digits
        self.stopwords_set = set(stopwords.words('english'))
        self.wv = None
        self.model = None
        self.get_saved_model_if_exists()

    def is_trained(self):
        return self.wv is not None

    def get_save_path(self):
        sw_flag = "with_sw" if self.include_sw else "no_sw"
        model_type = "skipgram" if self.sg == 1 else "cbow"
        digits_flag = "with_digits" if self.include_digits else "no_digits"
        filename = f"word2vec_{model_type}_v{self.vector_size}_w{self.window}_min{self.min_count}_{sw_flag}_{digits_flag}.kv"
        return os.path.join(self.save_dir, filename)

    def get_tensorboard_log_dir(self):
        sw_flag = "with_sw" if self.include_sw else "no_sw"
        model_type = "skipgram" if self.sg == 1 else "cbow"
        digits_flag = "with_digits" if self.include_digits else "no_digits"
        base_log_dir = "tensorboard_logs"
        log_dir_name = f"word2vec_{model_type}_v{self.vector_size}_w{self.window}_min{self.min_count}_{sw_flag}_{digits_flag}"
        return os.path.join(base_log_dir, log_dir_name)

    def get_saved_model_if_exists(self):
        save_path = self.get_save_path()
        if os.path.exists(save_path):
            print(f"  Loading trained embeddings from {save_path}")
            try:
                self.wv = gensim.models.KeyedVectors.load(save_path)
            except Exception as e:
                print(f"  Error loading embeddings from {save_path}: {e}. Will attempt to retrain if needed.")
                self.wv = None 
        else: print(f"  No saved embeddings found at {save_path}")

    def save_model(self):
        if self.wv:
            save_path = self.get_save_path()
            try:
                self.wv.save(save_path)
                print(f"  Saved word embeddings to {save_path}")
            except Exception as e:
                print(f"  Error saving word embeddings to {save_path}: {e}")
        else: print("  Warning: Word embeddings not trained/loaded, nothing to save.")

    def train(self, texts: List[str]):
        print("  Training Word2Vec model...")
        tokenized_texts = [text.split() for text in texts if text] 
        if not tokenized_texts:
            print("  Error: No non-empty texts provided for training. Skipping Word2Vec training.")
            return
        try:
            self.model = gensim.models.Word2Vec(
                sentences=tokenized_texts, vector_size=self.vector_size, window=self.window,
                min_count=self.min_count, workers=self.workers, sg=self.sg
            )
            self.wv = self.model.wv
            self.save_model()
        except Exception as e:
            print(f"  Error during Word2Vec training: {e}")
            self.wv = None #

    def save_for_tensorboard(self, politeness_association_map: Optional[Dict[str, str]] = None):
        """Saves embedding vectors and metadata for TensorBoard."""
        if self.wv is None:
            print("Error: Word embeddings (self.wv) not available. Cannot generate TensorBoard files.")
            return

        log_dir = self.get_tensorboard_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print(f"  Preparing TensorBoard files in: {log_dir}")

        vocab = self.wv.index_to_key

        weights_np = self.wv.vectors

        # --- Generate Metadata ---
        metadata_path = os.path.join(log_dir, 'metadata.tsv')
        print("    Generating metadata.tsv...")
        with open(metadata_path, "w", encoding='utf-8') as f:
            # Make sure header matches the simplified politeness categories
            f.write("Word\tPOS\tFrequency\tIsStopword\tPolarity\tPoliteness\n")
            for word in vocab:
                try:
                    tag = nltk.pos_tag([word])[0][1]
                except Exception:
                    tag = "UNK" # Handle potential errors during POS tagging
                try:
                    if hasattr(self.wv, 'get_vecattr'):
                        freq = self.wv.get_vecattr(word, 'count')
                    elif self.model and word in self.model.wv.key_to_index:
                         idx = self.model.wv.key_to_index[word]
                         freq = self.model.wv.expandos['count'][idx] 
                    else:
                        freq = 0 
                except (AttributeError, KeyError, IndexError):
                    freq = 0 
                is_stop = word in self.stopwords_set
                try:
                    polarity = TextBlob(word).sentiment.polarity
                except Exception:
                    polarity = 0.0 
                politeness_label = "N/A"
                if politeness_association_map:
                    politeness_label = politeness_association_map.get(word, "Not_Found_In_Stats") 
                f.write(f"{word}\t{tag}\t{freq}\t{is_stop}\t{polarity:.2f}\t{politeness_label}\n")
        print(f"    Metadata saved to {metadata_path}")

        # --- Save Checkpoint ---
        print("    Saving embeddings checkpoint...")
        try:
            weights_tf = tf.Variable(weights_np, name="word_embeddings")
            checkpoint = tf.train.Checkpoint(embedding=weights_tf)
            checkpoint_prefix = os.path.join(log_dir, "embedding.ckpt")
            checkpoint.save(checkpoint_prefix)
            print(f"    Checkpoint saved with prefix: {checkpoint_prefix}")
        except Exception as e:
             print(f"Error saving TensorFlow checkpoint: {e}")
             return 

        # --- Configure Projector ---
        print("    Configuring TensorBoard projector...")
        try:
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE" 
            embedding.metadata_path = 'metadata.tsv' 
            projector.visualize_embeddings(log_dir, config)
            print(f"    Projector config saved in {log_dir}")
        except Exception as e:
            print(f"Error configuring TensorBoard projector: {e}")
            return # Stop if projector config fails


        # --- Final Instructions ---
        parent_log_dir = os.path.dirname(log_dir)
        print("\nTensorBoard generation complete.")
        print("To view:")
        print(f"  1. Navigate to the project directory in your terminal.")
        print(f"  2. Run: tensorboard --logdir {os.path.abspath(parent_log_dir)}")
        print(f"  3. Open the generated URL in your browser and go to the 'Projector' tab.")
        print(f"  4. Select the run '{os.path.basename(log_dir)}'.")
        print(f"  5. Under 'Label by', select 'Word'. Under 'Color by', select 'Politeness'.")


def calculate_politeness_association(train_texts: List[str], train_labels: List[str],
                                    vocabulary: Set[str]) -> Dict[str, str]:
    """
    Calculates dominant politeness category for words in vocabulary.
    Simplified: Removes 'Rare' prefixes and assigns ties to 'Neutral'.
    """
    print("  Calculating simplified politeness association...")
    unique_labels = sorted(list(set(train_labels)))

    expected_labels = ['impolite', 'neutral', 'somewhat polite', 'polite']
    word_counts_by_label = {label: Counter() for label in expected_labels}

    print(f"  Processing {len(train_texts)} texts with labels...")
    processed_count = 0
    for text, label in zip(train_texts, train_labels):
        normalized_label = label.lower().strip()
        if normalized_label not in word_counts_by_label:
            print(f"Warning: Unexpected label '{label}' found. Skipping.") 
            continue

        tokens = text.split()
        for token in tokens:
            if token in vocabulary:
                word_counts_by_label[normalized_label][token] += 1
        processed_count += 1
        if processed_count % 10000 == 0:
             print(f"    Processed {processed_count}/{len(train_texts)} samples for stats...")


    politeness_assoc = {}
    label_order = ['impolite', 'neutral', 'somewhat polite', 'polite']

    print(f"  Assigning politeness association to {len(vocabulary)} vocabulary words...")
    processed_vocab_count = 0
    for word in vocabulary:
        max_count = -1
        dominant_labels = [] 
        total_count = 0

        for label in label_order:
            if label in word_counts_by_label:
                count = word_counts_by_label[label].get(word, 0)
                total_count += count
                if count > max_count:
                    max_count = count
                    dominant_labels = [label] 
                elif count == max_count and count > 0: # Found a tie with the current max
                    dominant_labels.append(label)

        # Determine final label
        final_label = "Unknown" # Default
        if total_count == 0:
            final_label = "Not_Seen_In_Labeled_Data" 
        elif len(dominant_labels) > 1:
            final_label =  "neutral" # Assign to neutral if there's a tie
        elif len(dominant_labels) == 1:
            final_label = dominant_labels[0] # Assign the single dominant label

        politeness_assoc[word] = final_label
        processed_vocab_count += 1
        if processed_vocab_count % 1000 == 0:
            print(f"    Assigned labels to {processed_vocab_count}/{len(vocabulary)} words...")

    print(f"  Simplified politeness association calculated for {len(vocabulary)} words.")
    final_label_counts = Counter(politeness_assoc.values())
    print("  Final Politeness Label Distribution:")
    for label, count in final_label_counts.most_common():
        print(f"    {label}: {count}")

    return politeness_assoc


def setup_and_get_embeddings(config: Dict[str, Any], data_loader: DataLoader,
                             train_files: List[str], feature_col: str, target_col: str) -> Optional[WordEmbeddingsModel]:
    """Initializes, trains, or loads the WordEmbeddingsModel."""
    print("Setting up embedding model...")
    embedding_model = WordEmbeddingsModel(**config)

    if not embedding_model.is_trained():
        print("  Embeddings not found or configured differently, training required.")
        train_texts_for_w2v, _ = data_loader.process_in_batches(
            train_files, feature_col, target_col, 
            include_sw=config['include_sw'],
            include_digits=config['include_digits']
        )
        if not train_texts_for_w2v:
             print("Error: No training data loaded. Cannot train embeddings.")
             return None
        embedding_model.train(train_texts_for_w2v)
        if not embedding_model.is_trained():
             print("Error: Training finished but model is still not marked as trained. Check training logs.")
             return None
    else:
        print("  Loaded existing embeddings.")

    return embedding_model


# --- Main Execution Logic ---
if __name__ == '__main__':
    print("Starting TensorBoard Generation Script...")
    download_nltk_resources()

    # --- Configuration ---
    TARGET_INCLUDE_DIGITS = False
    TARGET_INCLUDE_SW = True
    TARGET_VECTOR_SIZE = 200
    TARGET_WINDOW = 10
    TARGET_MIN_COUNT = 2
    TARGET_SG = 1 # 1 for skip-gram, 0 for CBOW

    DATA_DIR = "data"
    FEATURE_COL = 'text'
    TARGET_COL = 'label' # Make sure this matches the CSV header exactly

    train_cot_path = os.path.join(DATA_DIR, 'train', 'train_cot.csv')
    train_few_shot_path = os.path.join(DATA_DIR, 'train', 'train_few_shot.csv')

    file_paths = {
        'train_cot': train_cot_path,
        'train_few_shot': train_few_shot_path,
    }

    train_files_list = []
    if os.path.exists(train_cot_path):
        train_files_list.append(train_cot_path)
    else: print(f"Warning: Train file not found at {train_cot_path}")
    if os.path.exists(train_few_shot_path):
        train_files_list.append(train_few_shot_path)
    else: print(f"Warning: Train file not found at {train_few_shot_path}")

    if not train_files_list:
        print("Error: No training files found. Exiting.")
        exit()

    embedding_config = {
        "vector_size": TARGET_VECTOR_SIZE, "window": TARGET_WINDOW,
        "min_count": TARGET_MIN_COUNT, "workers": os.cpu_count() or 4,
        "sg": TARGET_SG, "include_sw": TARGET_INCLUDE_SW,
        "include_digits": TARGET_INCLUDE_DIGITS
    }

    # --- Initialize Components ---
    text_preprocessor = TextPreprocessor()
    data_loader = DataLoader({k: v for k, v in file_paths.items() if os.path.exists(v)}, text_preprocessor)

    # --- Load Training Data (for politeness stats) ---
    print("Loading training data for politeness calculation...")
    train_texts_for_stats, train_labels_for_stats = data_loader.process_in_batches(
        train_files_list, FEATURE_COL, TARGET_COL,
        include_sw=TARGET_INCLUDE_SW, 
        include_digits=TARGET_INCLUDE_DIGITS, 
        batch_size=4000
    )
    if not train_texts_for_stats:
        print("Error: Failed to load training data for statistics. Exiting.")
        exit()
    print(f"Loaded {len(train_texts_for_stats)} training samples for stats calculation.")


    # --- Setup/Train/Load Embeddings ---
    embedding_model = setup_and_get_embeddings(
        embedding_config, data_loader, train_files_list, FEATURE_COL, TARGET_COL
    )

    # --- Calculate Politeness Stats & Save ---
    if embedding_model and embedding_model.is_trained():
        if embedding_model.wv:
            w2v_vocabulary = set(embedding_model.wv.index_to_key)
            if not w2v_vocabulary:
                 print("Error: Word2Vec vocabulary is empty after loading/training. Cannot proceed.")
            else:
                print(f"Calculating politeness stats for {len(w2v_vocabulary)} words in vocabulary...")
                politeness_stats = calculate_politeness_association(
                    train_texts_for_stats,
                    train_labels_for_stats,
                    w2v_vocabulary
                )
                embedding_model.save_for_tensorboard(politeness_association_map=politeness_stats)
        else:
            print("Error: Embedding model wv attribute is None after setup. Cannot generate TensorBoard files.")
    elif embedding_model and not embedding_model.is_trained():
         print("Error: Embedding model setup completed, but model is not trained (possibly due to errors). Cannot generate TensorBoard files.")
    else: 
        print("Error: Embedding model could not be loaded or trained. Cannot generate TensorBoard files.")

    print("Script finished.")