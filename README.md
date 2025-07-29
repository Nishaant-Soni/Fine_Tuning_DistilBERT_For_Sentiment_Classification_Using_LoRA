# Fine-Tuning DistilBERT For Sentiment Classification Using Low Rank Adaption (LoRA)

This project demonstrates how to fine-tune a pre-trained language model using Low-Rank Adaptation (LoRA) for a sentiment classification task. The project uses the `distilbert-base-uncased` model from Hugging Face's Transformers library and fine-tunes it on a truncated version of the IMDb dataset for sentiment analysis.

## Table of Contents
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project is a truncated version of the IMDb dataset, which contains movie reviews labeled as either positive or negative.

- **Source**: [shawhin/imdb-truncated](https://huggingface.co/datasets/shawhin/imdb-truncated)

## Model

The model used is `distilbert-base-uncased`, a smaller and faster version of BERT. The model is fine-tuned using the Low-Rank Adaptation (LoRA) technique, which reduces the number of trainable parameters and speeds up the training process.

## Training

The training process involves the following steps:

1. **Data Loading**: Load the IMDb dataset.
2. **Tokenization**: Tokenize the text data using the `distilbert-base-uncased` tokenizer.
3. **Model Preparation**: Load the pre-trained `distilbert-base-uncased` model and configure it for sequence classification.
4. **LoRA Configuration**: Apply LoRA to the model to reduce the number of trainable parameters.
5. **Training**: Train the model using the `Trainer` class from the Transformers library.

## Evaluation

The model is evaluated on the validation set using multiple metrics, including:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: Evaluates the proportion of true positive predictions among all positive predictions.
- **Recall**: Measures the proportion of true positive predictions among all actual positive instances.
- **F1-Score**: Provides a harmonic mean of precision and recall.

The evaluation function computes these metrics to assess the model's performance.

## Results

The trained model achieves the following results on the validation set:

- **Accuracy**: 90.80%
- **Precision**: 91.63%
- **Recall**: 89.80%
- **F1-Score**: 90.71%

## Usage

To use the trained model for predictions, follow these steps:

1. Load the trained model and tokenizer:
    ```python
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_checkpoint = "distilbert-base-uncased-lora-text-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    ```

2. Make predictions on new text data:
    ```python
    text_list = ["It was good.", "Not a fan, don't recommend.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]

    for text in text_list:
        inputs = tokenizer.encode(text, return_tensors='pt')
        logits = model(inputs).logits
        predictions = torch.argmax(logits, 1)
        print(text + " - " + id2label[predictions.tolist()[0]])
    ```

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [IMDb Dataset](https://huggingface.co/datasets/shawhin/imdb-truncated)

This project was developed as part of a learning exercise in fine-tuning language models using advanced techniques.
