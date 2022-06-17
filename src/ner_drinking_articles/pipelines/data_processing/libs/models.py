import csv
import numpy as np
import streamlit as st
from scipy.special import softmax
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
import urllib

# Procedures to load the models once

# NER MODELS


# Start bert model
@st.experimental_singleton
def bert():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer)


# Start flair
@st.experimental_singleton
def flair():
    return SequenceTagger.load("flair/ner-english-large")


# Start roberta
@st.experimental_singleton
def roberta():
    tokenizer = AutoTokenizer.from_pretrained(
        "Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained(
        "Jean-Baptiste/roberta-large-ner-english")
    return pipeline('ner', model=model, tokenizer=tokenizer,
                    aggregation_strategy="simple")

# SA MODELS

# Start Roberta Sentiment
def robertaSentiment():
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest")

    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    labels = ['negative', 'neutral', 'positive']
    return model, tokenizer, labels

def newsmtscSentiment():
    tokenizer = AutoTokenizer.from_pretrained("RogerKam/roberta_fine_tuned_sentiment_newsmtsc")

    model = AutoModelForSequenceClassification.from_pretrained("RogerKam/roberta_fine_tuned_sentiment_newsmtsc")

    labels = ['negative', 'neutral', 'positive']
    return model, tokenizer, labels

# Procedures to exploit the models


# Process bert
@st.experimental_memo
def procBert(articles):
    model = bert()

    globalResults = []
    for i, contents in enumerate(articles):

        results = model(contents)
        globalResults.append(results)
    return globalResults


# Process roberta
@st.experimental_memo
def procRoberta(articles):
    model = roberta()

    globalResults = []
    for i, contents in enumerate(articles):

        results = model(contents)
        globalResults.append(results)
    return globalResults


# Process flair
@st.experimental_memo
def procFlair(articles):
    model = flair()
    globalResults = []
    for article in articles:
        for text in article:
            articleResult = []
            for t in text:
                # st.write(t)
                contents = Sentence(t)
                model.predict(contents)
                currentResult = []
                for r in contents.get_spans('ner'):
                    if [r.text, r.get_label("ner").value] not in currentResult:
                        currentResult.append(
                            [r.text, r.get_label("ner").value])
                articleResult.append(currentResult)
        globalResults.append(articleResult)
    return globalResults


# Process Sentiment Analysis
# @st.experimental_memo
def procSentiment(articles):
    if st.session_state.saModel == 'newsmtsc':
        model, tokenizer, labels = newsmtscSentiment()
    else:
        model, tokenizer, labels = robertaSentiment()

    globalResults = []
    globalLabelPerSentence = []

    for article in articles:
        for text in article:
            articleResults = {}
            labelPerSentence = []
            for lab in labels:
                articleResults[lab] = 0
            for contents in text:
                # st.write(contents)
                # Transformers cannot handle texts longer than 512 tokens for
                # Sentiment Analysis...
                # Compensate that with a magic trick (split the text,
                # do the sentiment analysis for each split, and avg it all)
                tokens = tokenizer.encode_plus(contents,                 
                                               add_special_tokens=False,
                                               return_tensors='pt')

                chunk = 512
                input_id_chunks = list(tokens['input_ids'][0].split(chunk - 2))
                mask_chunks = list(
                    tokens['attention_mask'][0].split(chunk - 2))

                for c in range(len(input_id_chunks)):
                    input_id_chunks[c] = torch.cat(
                        [
                            torch.tensor([101]),
                            input_id_chunks[c],
                            torch.tensor([102])
                        ]
                    )

                    mask_chunks[c] = torch.cat(
                        [
                            torch.tensor([1]),
                            mask_chunks[c],
                            torch.tensor([1])
                        ]
                    )

                    padding = chunk - input_id_chunks[c].shape[0]
                    if padding > 0:
                        input_id_chunks[c] = torch.cat(
                            [
                                input_id_chunks[c],
                                torch.Tensor([0] * padding),
                            ]
                        )

                        mask_chunks[c] = torch.cat(
                            [
                                mask_chunks[c],
                                torch.Tensor([0] * padding),
                            ]
                        )

                    assert(len(input_id_chunks[c]) == chunk)

                input_ids = torch.stack(input_id_chunks)
                attention_mask = torch.stack(mask_chunks)

                input_dict = {
                    'input_ids': input_ids.long(),
                    'attention_mask': attention_mask.int()
                }

                outputs = model(**input_dict)
                # result = torch.nn.functional.softmax(outputs[0], dim=-1)
                # result = result.mean(dim=0)

                scores = outputs[0][0].detach().numpy()
                scores = softmax(scores)

                ranking = np.argsort(scores)
                ranking = ranking[::-1]

                labelPerSentence.append(labels[ranking[0]])

                for i in range(scores.shape[0]):
                    articleResults[labels[ranking[i]]] += scores[ranking[i]]
            for lab in labels:
                articleResults[lab] /= len(text)
            globalResults.append(articleResults)
            globalLabelPerSentence.append(labelPerSentence)
    return globalResults, globalLabelPerSentence
