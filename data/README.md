---
license: cdla-permissive-2.0
task_categories:
- text-classification
language:
- en
size_categories:
- 100K<n<1M
tags:
- synthetic
- NLP
- politeness
- benchmark
- few-shot
- chain-of-thought
---
# Polite Guard

- **Dataset type**: Synthetic and Annotated
- **Task**: Text Classification
- **Domain**: Classification of text into polite, somewhat polite, neutral, and impolite categories
- **Source Code**: (https://github.com/intel/polite-guard)
- **Model**: (https://huggingface.co/Intel/polite-guard)

This dataset is for [**Polite Guard**](https://huggingface.co/Intel/polite-guard): an open-source NLP language model developed by Intel, fine-tuned from BERT for text classification tasks. Polite Guard is designed to classify text into four categories: polite, somewhat polite, neutral, and impolite. The model, along with its accompanying datasets and [source code](https://github.com/intel/polite-guard), is available on Hugging Face* and GitHub* to enable both communities to contribute to developing more sophisticated and context-aware AI systems.

## Use Cases

Polite Guard provides a scalable model development pipeline and methodology, making it easier for developers to create and fine-tune their own models. Other contributions of the project include:

- **Improved Robustness**:
Polite Guard enhances the resilience of systems by providing a defense mechanism against adversarial attacks. This ensures that the model can maintain its performance and reliability even when faced with potentially harmful inputs.
- **Benchmarking and Evaluation**:
The project introduces the first politeness benchmark, allowing developers to evaluate and compare the performance of their models in terms of politeness classification. This helps in setting a standard for future developments in this area.
- **Enhanced Customer Experience**:
By ensuring respectful and polite interactions on various platforms, Polite Guard can significantly boost customer satisfaction and loyalty. This is particularly beneficial for customer service applications where maintaining a positive tone is crucial.

## Dataset Description

The dataset consists of three main components:

- 50,000 samples generated using *Few-Shot prompting*
- 50,000 samples generated using *Chain-of-Thought (CoT) prompting*
- 200 *annotated* samples from corporate trainings with the personal identifiers removed

The synthetic data is split into training (80%), validation (10%), and test (10%) sets, with each set balanced according to the label. The real annotated data is used solely for evaluation purposes.

Each example contains:

- **text**: The text input (string)
- **label**: The classification label (category: polite, somewhat polite, neutral, and impolite)
- **source**: The language model used to generate synthetic text and LMS (Learning Management Systems) for corporate trainings (category)
- **reasoning**: The reasoning provided by the language model for generating text that aligns with the specified label and category (string)

The synthetic data consists of customer service interactions across various sectors, including finance, travel, food and drink, retail, sports clubs, culture and education, and professional development. To ensure *data regularization*, the labels and categories were randomly selected, and a language model was instructed to generate synthetic data based on the specified categories and labels. To ensure *data diversity*, the generation process utilized multiple prompts and the large language models listed below.

- [Llama 3.1 8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Gemma 2 9B-It](https://huggingface.co/google/gemma-2-9b-it)
- [Mixtral 8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

The code for the data generator pipeline is available [here](https://github.com/intel/polite-guard). For more details on the prompts used and the development of the generator, refer to this [article](https://medium.com/p/0ff98eb226a1).

## Description of labels

- **polite**: Text is considerate and shows respect and good manners, often including courteous phrases and a friendly tone.
- **somewhat polite**: Text is generally respectful but lacks warmth or formality, communicating with a decent level of courtesy.
- **neutral**: Text is straightforward and factual, without emotional undertones or specific attempts at politeness.
- **impolite**: Text is disrespectful or rude, often blunt or dismissive, showing a lack of consideration for the recipient's feelings.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Intel/polite-guard")
```

## Articles

To learn more about the implementation of the data generator and fine-tuner packages, refer to

- [Synthetic Data Generation with Language Models: A Practical Guide](https://medium.com/p/0ff98eb226a1), and
- [How to Fine-Tune Language Models: First Principles to Scalable Performance](https://medium.com/p/78f42b02f112).

For more AI development how-to content, visit [Intel® AI Development Resources](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html).

## Join the Community

If you are interested in exploring other models, join us in the Intel and Hugging Face communities. These models simplify the development and adoption of Generative AI solutions, while fostering innovation among developers worldwide. If you find this project valuable, please like ❤️ it on Hugging Face and share it with your network. Your support helps us grow the community and reach more contributors.

## Disclaimer

Polite Guard has been trained and validated on a limited set of data that pertains to customer reviews, product reviews, and corporate communications. Accuracy metrics cannot be guaranteed outside these narrow use cases, and therefore this tool should be validated within the specific context of use for which it might be deployed. This tool is not intended to be used to evaluate employee performance. This tool is not sufficient to prevent harm in many contexts, and additional tools and techniques should be employed in any sensitive use case where impolite speech may cause harm to individuals, communities, or society.