# NER Drinking Articles - Proof Of Concept

## Introduction 
This POC demonstrates a pipeline that can be used to create a knowledge database from news articles. The dataset for our use case is extracted from websites specializing in bartending-related subjects, but any topic should work.

It does the following tasks to extract information:

- News scraping from a set of pre-defined URLs
- Data loading
- Data pre-processing (coreference with the spacy-provided crosslingual_coreference and en_core_web_sm, and sentence splitting with nltk's punkt tokenizer)
- Sentiment analysis over each sentence using Twitter RoBERTa, and extraction of per-sentence and per-article metrics
- NER using the Flair model + basic relationship extraction over each sentence
- Sense disambiguation for each entity with addition of context and definition when available, and association of sentiments detected for all sentences containing the entity processed
- Graphical reporting with visualizations, illustrating the different relationships in detail
- Data output on JSON files (articles catalog + entities catalog)

## Getting Started

### How to install dependencies

To install them, run:

```
pip install -r requirements.txt
```

You can also isolate the application by using a virtual environment.

For Linux-based & Mac OS:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### How to run the application

You can run the streamlit project with:

```
streamlit run main.py
```

The main application automatically handles the creation of a Kedro Session when manually executing the pipeline.
This is done by selecting at least one URL, and clicking the "Execute pipeline" button.

## Visualize the pipeline

Using Kedro Viz, you can generate a web-based visualization. 

It shows the pipeline with its nodes, and the interactions between each component.

You can start the server with:

```
kedro viz
```
