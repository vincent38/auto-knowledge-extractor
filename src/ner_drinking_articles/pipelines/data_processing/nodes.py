"""
This file contains the nodes for the pipeline 'data_processing'
generated using Kedro 0.18.1
"""
from .libs import models, elink
from newsplease import NewsPlease
from kedro.extras.datasets.json import JSONDataSet
import spacy
import streamlit as st
from streamlit_agraph import Node, Edge
import crosslingual_coreference
import itertools
from fuzzywuzzy import fuzz
import nltk
nltk.download("punkt")


# Data scrapper - A dissocier de la pipeline


def scrap_data():
    st.header("Data scrapping")
    nodesFound = []
    articlesDict = []
    st.session_state.sankeyLabels = []
    st.session_state.sankeySrc = []
    st.session_state.sankeyTarget = []
    rawArticles = JSONDataSet(filepath="data/01_raw/articles.json")
    try:
        with st.spinner("Gathering test data..."):
            data = []
            # for url in articles.LIST_URLs:
            for id, url in enumerate(st.session_state.urlArticles):
                article = NewsPlease.from_url(url)
                # Get the article publish or modify date
                if article.get_dict()['date_modify'] != None:
                    date = article.get_dict()['date_modify']
                else:
                    date = article.get_dict()['date_publish']
                # Remove some unwanted written forms
                text = article.get_dict()['maintext']
                text = text.replace("'s", "")
                data.append((text))
                st.write("Gathered data from URL:", url)
                nodesFound.append(Node(id=str(id),
                                  label=url,
                                  size=400))
                st.session_state.sankeyLabels.append(url)
                # Create an entry for the knowledge DB
                entry = {
                    'id': id,
                    'url': url,
                    'day': date.day,
                    'month': date.month,
                    'year': date.year,
                    'text': "",
                    'sentiments': []
                }
                articlesDict.append(entry)
            rawArticles.save(data)
        st.success("Successfully downloaded test data.")
        # nodes.save(nodesFound)
    except Exception as e:
        st.warning("Could not download articles. Reason:"+str(e))
    st.session_state.articleNodes = nodesFound
    # st.session_state.articlesDict = articlesDict
    return rawArticles, articlesDict


# Extract data from the json file
def load_data(rawArticles):
    st.header("Data loading")
    with st.spinner("Getting data from the DataSet..."):
        data = rawArticles.load()
    st.success("Successfully loaded data from dataset.")
    return data


# Coreference node
def do_coreference(articles, articlesDict):
    st.header("Coreference processing")
    with st.spinner("Loading coreference model..."):
        coref = spacy.load('en_core_web_sm',
                           disable=['ner', 'tagger', 'parser',
                                    'attribute_ruler', 'lemmatizer'])
        coref.add_pipe("xx_coref",
                       config={"chunk_size": 2500, "chunk_overlap": 2,
                               "device": -1})
    with st.spinner("Processing coreference model over the texts..."):
        corefArticles = []
        for id, text in enumerate(articles):
            # corefText = coref(text)._.resolved_text.split('.')
            corefText = nltk.tokenize.sent_tokenize(
                coref(text)._.resolved_text
            )
            if (corefText[0].split(' ')[0] == "News"):
                corefText[0] = corefText[0].split(' ', 4)[4]
            corefArticles.append([corefText])
            articlesDict[id]['text'] = corefText
    st.success("Successfully processed coreference task.")
    return corefArticles, articlesDict


# Prepare print node
def print_all_texts(articles):
    st.header("Texts in test dataset")
    for text in articles:
        st.write(text)


# NER Node
def process_ner(articles):
    st.header("Process NER")
    with st.spinner("Processing NER over texts..."):
        ners = models.procFlair(articles)
    st.success("Successfully processed NER task.")
    return ners


# Prepare print NER node
def process_sense_disambiguation(ner, sesa):
    st.header("NERs found in texts")
    atoeEdges = []
    etoeEdges = []
    etosEdges = []
    nodesFound = []
    entitiesDict = []
    with st.spinner("Processing Entity Sense Disambiguation task..."):
        for i, articleList in enumerate(ner):
            # For each article
            with st.expander("Article "+str(i)):
                for sid, text in enumerate(articleList):
                    # For each sentence in an article
                    sentenceSentiment = sesa[i][sid]
                    nodesInSentence = []
                    for e in text:
                        # For each element in a sentence

                        # Fuzzy matching - will take an existing element if present
                        matchElement = e[0]
                        matchScore = 90
                        for node in nodesFound:
                            fuzzScore = fuzz.ratio(e[0], node.id)
                            print(fuzzScore, matchScore)
                            if fuzzScore >= matchScore:
                                matchScore = fuzzScore
                                matchElement = node.id
                        print("Initial entity:", e[0],
                              "New entity:", matchElement)
                        # At the end, fmatch outputs the best node id possible

                        entity = {
                            'entity': matchElement,
                            'type': e[1],
                            'sentenceSentiments': [],
                            'duck': "",
                            'wikidata': "",
                            'maps': "",
                            'isIn': []
                        }

                        # Fuzzy matching done, proceed
                        if e[1] == 'LOC':
                            color = st.session_state.colorLocNode
                        elif e[1] == 'ORG':
                            color = st.session_state.colorOrgNode
                        elif e[1] == 'MISC':
                            color = st.session_state.colorMiscNode
                        else:
                            color = st.session_state.colorPerNode
                        n = Node(id=matchElement, label=matchElement,
                                 size=400, color=color)
                        if (matchElement, sentenceSentiment) not in [(edg.source,
                                                                      edg.target)
                                                                     for edg in
                                                                     etosEdges]:
                            etosEdges.append(Edge(source=matchElement,
                                                  label="has_sentiment",
                                                  target=sentenceSentiment,
                                                  color='#9C9C9C',
                                                  type="STRAIGHT"))
                            entity['sentenceSentiments'].append(sentenceSentiment)
                        if matchElement not in [no.id for no in nodesInSentence]:
                            nodesInSentence.append(n)
                        if matchElement not in [no.id for no in nodesFound]:
                            # Append in list of known nodes and print its
                            # detailed information, just this once
                            nodesFound.append(n)
                            st.session_state.entitiesDict[matchElement] = {}
                            st.session_state.entitiesDict[matchElement]['isIn'] = []
                            st.session_state.sankeyLabels.append(matchElement)
                            wikidataID = elink.find_wikidata_id(
                                str(matchElement))
                            duckData = elink.find_duck_def(str(matchElement))

                            entity['duck'] = duckData
                            entity['wikidata'] = 'https://www.wikidata.org/wiki/' + wikidataID
                            entity['maps'] = 'https://www.google.com/maps/search/?api=1&query=' + str(matchElement).replace(' ', '%20')

                            st.session_state.entitiesDict[matchElement]['type'] = e[1]

                            st.session_state.entitiesDict[matchElement]['sentiment'] = sentenceSentiment

                            if wikidataID != 'id-less':
                                st.session_state.entitiesDict[matchElement][
                                    'wikidata'] = 'https://www.wikidata.org/wiki/' + wikidataID

                            st.session_state.entitiesDict[matchElement]['gmaps'] = 'https://www.google.com/maps/search/?api=1&query=' + str(
                                matchElement).replace(' ', '%20')

                            if duckData != 'no-data' and duckData != "":
                                st.session_state.entitiesDict[matchElement]['duckIA'] = duckData

                        if (str(i), matchElement) not in [(edg.source, edg.target)
                                                          for edg in atoeEdges]:
                            # Define link between article and entity
                            atoeEdges.append(Edge(source=str(i),
                                                  label="has_entity",
                                                  target=matchElement,
                                                  type="STRAIGHT"))
                            st.session_state.sankeySrc.append(
                                st.session_state.sankeyLabels.index(
                                    st.session_state.urlArticles[i]
                                )
                            )
                            st.session_state.sankeyTarget.append(
                                st.session_state.sankeyLabels.index(
                                    matchElement)
                            )
                            entity['isIn'] = st.session_state.urlArticles[i]
                            st.session_state.entitiesDict[matchElement]['isIn'].append(st.session_state.urlArticles[i])
                            entitiesDict.append(entity)

                        # st.markdown(sentence)
                        # Add the relationship edges between
                        # each node in the current sentence
                        for a, b in itertools.combinations(nodesInSentence, 2):
                            if (a.to_dict()['id'], b.to_dict()['id']) not in [
                                    (edg.source, edg.target) for edg in etoeEdges]:
                                etoeEdges.append(Edge(source=a.to_dict()['id'],
                                                      label="are_linked_together",
                                                      target=b.to_dict()['id'],
                                                      type="STRAIGHT"))
        st.session_state.entityNodes = nodesFound
        st.session_state.atoeEdges = atoeEdges
        st.session_state.etoeEdges = etoeEdges
        st.session_state.etosEdges = etosEdges
    st.success("Successfully processed entity sense disambiguation task.")
    return entitiesDict


# Node relationships

# Sentiment Analysis Node
def process_sentiment_analysis(articles, articlesDict):
    nodesFound = []
    atosEdges = []
    nodesFound.append(Node(id="neutral", label="Neutral 😐",
                      color="#FFA500", size=400))
    nodesFound.append(Node(id="positive", label="Positive 🙂",
                      color="#00FF00", size=400))
    nodesFound.append(Node(id="negative", label="Negative 🙁",
                      color="#FF0000", size=400))
    st.header("Process Sentiment Analysis")
    if st.session_state.saModel == 'newsmtsc':
        st.info("Using NewsMTSC model for the SA task.")
    else:
        st.info("Using fallback Twitter RoBERTa model for the SA task.")
    with st.spinner("Processing Sentiment Analysis over texts..."):
        sa, sesa = models.procSentiment(articles)
        for i, current in enumerate(sa):
            with st.expander("Article "+str(i)):

                # Put in articles dictionnary
                articlesDict[i]['sentiments'] = current

                # Check which field is biggest
                maxKey = max(current, key=current.get)

                if maxKey == "neutral":
                    atosEdges.append(Edge(source=str(i),
                                          label="is_of_sentiment",
                                          target="neutral",
                                          type="STRAIGHT"))
                if maxKey == "positive":
                    atosEdges.append(Edge(source=str(i),
                                          label="is_of_sentiment",
                                          target="positive",
                                          type="STRAIGHT"))
                if maxKey == "negative":
                    atosEdges.append(Edge(source=str(i),
                                          label="is_of_sentiment",
                                          target="negative",
                                          type="STRAIGHT"))

    st.session_state.sentimentsList = sa
    st.success("Successfully processed Sentiment Analysis")
    st.session_state.sentimentNodes = nodesFound
    st.session_state.atosEdges = atosEdges
    return sesa, articlesDict

def save_datasets(articlesDict, entitiesDict):
    articlesCatalog = JSONDataSet(
        filepath="data/03_primary/articles_catalog.json"
    )
    entitiesCatalog = JSONDataSet(
        filepath="data/03_primary/entities_catalog.json"
    )
    articlesCatalog.save(articlesDict)
    entitiesCatalog.save(entitiesDict)
    return articlesCatalog, entitiesCatalog
