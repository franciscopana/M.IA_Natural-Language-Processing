# **Natural Language Processing - Polite Guard Dataset**

## **Group Members**
- Adriano Machado (202105352)  
- Félix Martins (202108837)  
- Francisco da Ana (202108762)  

## **Project Description**
This project develops and evaluates NLP classifiers for the Polite Guard dataset, a synthetic and annotated text classification dataset categorizing text into polite, somewhat polite, neutral, and impolite. Using traditional ML techniques (e.g., Naive Bayes, Logistic Regression, SVM) and excluding deep learning, we explore pre-processing, feature extraction, and sparse/dense representations like word embeddings. The project includes exploratory data analysis (EDA), feature engineering, classifier training, and evaluation with Precision, Recall, F1, and macro-F1 metrics, plus error analysis of top models. We build from a baseline, testing techniques to enhance performance within the dataset’s context.

---

## **Dataset Overview**

### **Source and Provenance**
The Polite Guard dataset is an open-source resource developed by Intel, fine-tuned from BERT, and made available on [GitHub](https://github.com/intel/polite-guard) and [Hugging Face](https://huggingface.co/Intel/polite-guard). It consists of:
- **50,000 synthetic samples** generated via Few-Shot prompting.
- **50,000 synthetic samples** generated via Chain-of-Thought (CoT) prompting.
- **200 annotated samples** from corporate training data (personal identifiers removed).

The synthetic data simulates customer service interactions across domains like finance, travel, food and drink, retail, sports clubs, culture and education, and professional development. It was generated using multiple large language models (Llama 3.1 8B-Instruct, Gemma 2 9B-It, Mixtral 8x7B-Instruct-v0.1) to ensure diversity, with prompts detailed in [this article](#).

### **Dataset Structure**
- **Training Set**: 80% of synthetic data (balanced across labels).
- **Validation Set**: 10% of synthetic data.
- **Test Set**: 10% of synthetic data.
- **Evaluation Set**: 200 real annotated samples (used solely for evaluation).

Each sample includes:
- **text**: The input text (string).
- **label**: One of *polite*, *somewhat polite*, *neutral*, or *impolite*.
- **source**: The model or system generating the text (e.g., LLM or LMS).
- **reasoning**: Explanation of why the text aligns with its label (for synthetic data).

### **Label Descriptions**
- **Polite**: Respectful, courteous, and friendly text.
- **Somewhat Polite**: Respectful but less warm or formal.
- **Neutral**: Factual and straightforward, lacking emotional tone.
- **Impolite**: Rude, blunt, or dismissive text.

---

## **Project Objectives**
As part of Assignment 1, we aim to:
1. Understand the dataset’s characteristics, provenance, and annotation process.
2. Conduct an exploratory data analysis (EDA) with visualizations (e.g., class distribution, word frequency, TF-IDF).
3. Apply pre-processing (e.g., tokenization, stopword removal) and feature extraction techniques (sparse features like Bag-of-Words or dense features like pre-trained word embeddings).
4. Train and evaluate traditional ML classifiers, reporting Precision, Recall, F1, and macro-F1 scores.
5. Establish a baseline model (e.g., Naive Bayes with basic features) and improve upon it through feature engineering, classifier selection, and hyperparameter tuning.
6. Perform error analysis on the best model to identify misclassification patterns.

---

## **References**
- Polite Guard GitHub: [https://github.com/intel/polite-guard](https://github.com/intel/polite-guard)
- Polite Guard Model: [https://huggingface.co/Intel/polite-guard](https://huggingface.co/Intel/polite-guard)

