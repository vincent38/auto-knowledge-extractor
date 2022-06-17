import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from kedro.runner import SequentialRunner
from streamlit_agraph import agraph, Config
import plotly.graph_objects as go
import plotly.express as px
import random
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


# Fixed set of URLs
LIST_URLs = [
    'https://www.thedrinksbusiness.com/2021/05/japanse-gin-ki-no-bi-hosts-pop-up-in-hong-kong/',
    'https://www.thespiritsbusiness.com/2022/04/havana-club-ceo-reveals-sustainability-goals/',
    'https://www.thespiritsbusiness.com/2022/04/pernod-ricard-launches-gtr-exclusive-cognac/',
    'https://www.thespiritsbusiness.com/2022/03/boxpark-names-pernod-ricard-pouring-partner/',
    'https://www.thedrinksbusiness.com/2022/04/joseph-castan-releases-crocodile-bottle-inspired-by-ancient-history/',
    'https://www.thedrinksbusiness.com/2022/04/uk-rum-brand-to-launch-the-first-rum-distilled-at-sea/',
    'https://www.thedrinksbusiness.com/2022/04/guinness-cold-brew-coffee-beer-launches-in-the-uk/',
    'https://www.thedrinksbusiness.com/2022/04/governor-of-new-york-legalises-drinks-to-go/',
    'https://www.thedrinksbusiness.com/2022/04/us-craft-beer-scene-grows-as-new-figures-and-top-50-breweries-revealed/',
    'https://www.thedrinksbusiness.com/2022/04/pernod-ricard-hails-very-strong-third-quarter-sales-increase/',
    'https://www.thedrinksbusiness.com/2022/04/bold-new-labels-ensure-pink-price-consistency/',
    'https://www.thedrinksbusiness.com/2022/01/pernod-ricard-korea-launches-responsible-drinking-campaign/',
    'https://www.thedrinksbusiness.com/2021/09/pernod-ricard-fy21-sales-bounce-back-ahead-of-fy19-levels/',
    'https://www.thedrinksbusiness.com/2022/06/la-gaffeliere-withdraws-from-saint-emilion-classification/',
    'https://www.just-drinks.com/news/pernod-ricard-updates-havana-club-3-anos-name-packaging/'
]

# Create a runner to run the pipeline
runner = SequentialRunner()

st.title('NER Drink Magazine experiment')

# Define session variables on first run + purge procedure
if 'articleNodes' not in st.session_state.keys():
    st.session_state.articleNodes = []

if 'entityNodes' not in st.session_state.keys():
    st.session_state.entityNodes = []

if 'sentimentNodes' not in st.session_state.keys():
    st.session_state.sentimentNodes = []

if 'entitiesDict' not in st.session_state.keys():
    st.session_state.entitiesDict = {}

if 'sentimentsList' not in st.session_state.keys():
    st.session_state.sentimentsList = []

if 'etoeEdges' not in st.session_state.keys():
    st.session_state.etoeEdges = []

if 'atoeEdges' not in st.session_state.keys():
    st.session_state.atoeEdges = []

if 'atosEdges' not in st.session_state.keys():
    st.session_state.atosEdges = []

if 'etosEdges' not in st.session_state.keys():
    st.session_state.etosEdges = []

if 'graphConf' not in st.session_state.keys():
    st.session_state.graphConf = []

if 'sankeyLabels' not in st.session_state.keys():
    st.session_state.sankeyLabels = []

if 'sankeySrc' not in st.session_state.keys():
    st.session_state.sankeySrc = []

if 'sankeyTarget' not in st.session_state.keys():
    st.session_state.sankeyTarget = []

if 'urlArticles' not in st.session_state.keys():
    st.session_state.urlArticles = []

if 'articlesDict' not in st.session_state.keys():
    st.session_state.articlesDict = []

if 'saModel' not in st.session_state.keys():
    st.session_state.saModel = ""

def purge_session_vars():
    st.session_state.articleNodes = []
    st.session_state.entityNodes = []
    st.session_state.sentimentNodes = []
    st.session_state.etoeEdges = []
    st.session_state.atoeEdges = []
    st.session_state.atosEdges = []
    st.session_state.etosEdges = []
    st.session_state.graphConf = []
    st.session_state.sankeyLabels = []
    st.session_state.sankeySrc = []
    st.session_state.sankeyTarget = []
    st.session_state.entitiesDict = {}
    st.session_state.sentimentsList = []
    st.session_state.urlArticles = []
    st.session_state.articlesDict = []
    st.session_state.saModel = ""
# End of definition step

# Options in sidebar
st.sidebar.header("Run options")

urlToProcess = st.sidebar.multiselect("Select articles \
                                    to process", LIST_URLs)

saModel = st.sidebar.selectbox(
    "Model to use for Sentiment Analysis",
    ('newsmtsc', 'twitter-roberta')
)

if st.sidebar.button("Execute pipeline"):
    purge_session_vars()
    st.session_state.urlArticles = urlToProcess
    st.session_state.saModel = saModel
    # Bootstrap the project - starts Kedro and loads all the needed stuff
    bootstrap_project(str(Path.cwd()))
    # Initialize a Kedro Session with the existing project
    with KedroSession.create("ner_drinking_articles", str(Path.cwd())) as session:
            # Run this session with our runner
            # The nodes uses streamlit to display information and 
            # store data, so seamless integration.
            print(session.run("dp", runner=runner))

# Graphical options

st.sidebar.header("Visualisation options")

st.session_state.colorLocNode = st.sidebar.color_picker(
    'Color for LOC entities',
    '#87A5F5'
)

st.session_state.colorOrgNode = st.sidebar.color_picker(
    'Color for ORG entities',
    '#A5FF92'
)

st.session_state.colorMiscNode = st.sidebar.color_picker(
    'Color for MISC entities',
    '#DA75FF'
)

st.session_state.colorPerNode = st.sidebar.color_picker(
    'Color for PER entities',
    '#8CF5ED'
)

# Memory management options

st.sidebar.header("Cache options")

if st.sidebar.button("Purge session variables"):
    purge_session_vars()

if st.sidebar.button("Purge Memo cache"):
    st.experimental_memo.clear()

if st.sidebar.button("Purge Singleton cache"):
    st.experimental_singleton.clear()

# Permanent report and visualizations from in-cache data
# Relies solely on session state variables to generate
# outputs and visualizations

if len(st.session_state.urlArticles) > 0:
    # We can print the report as we have content processed

    st.header('URLs processed in this report')

    for url in st.session_state.urlArticles:
        st.write(url)

    st.header("Relations between Entities in set of articles")

    entity_rel_config = Config(automaticRearrangeAfterDropNode=True,
                               key='etoe',
                               width=1500,
                               height=1000,
                               directed=False,
                               collapsible=True,
                               node={'labelProperty': 'label'},
                               link={'labelProperty': 'label',
                                     'renderLabel': False}
                               )

    # Options to show or hide relations
    nerRelNodes = st.session_state.entityNodes
    nerRelEdges = []

    if st.checkbox('Show relations between entities', value=True):
        nerRelEdges += st.session_state.etoeEdges

    if st.checkbox('Show sentiments linked to each element'):
        nerRelNodes += st.session_state.sentimentNodes
        nerRelEdges += st.session_state.etosEdges

    # Out Knowledge graph
    agraph(
        nodes=nerRelNodes,
        edges=nerRelEdges,
        config=entity_rel_config
    )

    st.header("Entities found in each article in dataset")

    # Use plotly here
    # Define colors of nodes

    colors = []

    for n in st.session_state.sankeyLabels:
        if n in st.session_state.entitiesDict:
            # Entity, set to color
            if st.session_state.entitiesDict[n]['type'] == 'LOC':
                colors.append(st.session_state.colorLocNode)
            elif st.session_state.entitiesDict[n]['type'] == 'ORG':
                colors.append(st.session_state.colorOrgNode)
            elif st.session_state.entitiesDict[n]['type'] == 'MISC':
                colors.append(st.session_state.colorMiscNode)
            else:
                colors.append(st.session_state.colorPerNode)
        else:
            colors.append("blue")

    colorsEdges = ["#"+''.join([random.choice('ABCDEF0123456789')
                               for i in range(6)])
                   for j in set(st.session_state.sankeySrc)]

    figure = go.Figure(
        data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="white", width=0.5),
                label=st.session_state.sankeyLabels,
                color=colors
            ),
            link=dict(
                source=st.session_state.sankeySrc,
                target=st.session_state.sankeyTarget,
                value=[1 for i in range(len(st.session_state.sankeySrc))],
                color=[colorsEdges[i] for i in st.session_state.sankeySrc]
            )
        )]
    )

    st.plotly_chart(figure, use_container_width=True)

    st.header("Sentiment analysis output for each article")

    sa_df = pd.DataFrame(
        data={
            'url': [url for url in st.session_state.urlArticles],
            'positive': [i['positive']*100
                         for i in st.session_state.sentimentsList],
            'negative': [i['negative']*100
                         for i in st.session_state.sentimentsList],
            'neutral': [i['neutral']*100
                        for i in st.session_state.sentimentsList],
            'dominant': [max(i, key=i.get)
                         for i in st.session_state.sentimentsList],
            'size': 1
        }
    )

    ternary = px.scatter_ternary(
        sa_df,
        a="positive",
        b="negative",
        c="neutral",
        hover_name="url",
        color="dominant",
        size="size",
        color_discrete_map={
            "positive": "green",
            "negative": "red",
            "neutral": "orange"
        }
    )

    st.plotly_chart(ternary)

    st.header("Detailed sentiment analysis")

    if st.session_state.saModel == 'newsmtsc':
        st.info("Computed using NewsMTSC model for the SA task.")
    else:
        st.info(
            "Computed using fallback Twitter RoBERTa model for the SA task."
        )

    for i, current in enumerate(st.session_state.sentimentsList):
        with st.expander("Article "+str(i)):

            # Check which field is biggest
            maxKey = max(current, key=current.get)

            if maxKey == "neutral":
                st.subheader("Article "+str(i)+" is mostly Neutral 😐")
            if maxKey == "positive":
                st.subheader("Article "+str(i)+" is mostly Positive 🙂")
            if maxKey == "negative":
                st.subheader("Article "+str(i)+" is mostly Negative 🙁")

            for k in current:
                st.metric(label=k,
                          value=str(round(float(current[k])*100, 2))+" %")

    # Lastly, print the sense disambiguation results

    st.header('Sense disambiguation results for each entity')

    entities = st.session_state.entitiesDict.items()

    for i, url in enumerate(st.session_state.urlArticles):
        with st.expander("Article "+str(i)):
            for id, entity in entities:
                # Check if entity is in selected article
                if url in entity["isIn"]:
                    st.subheader(id)
                    st.write('Lookup in Google Maps: ', entity['gmaps'])
                    if 'wikidata' in entity:
                        st.write('Lookup in WikiData: ', entity['wikidata'])
                    if 'duckIA' in entity:
                        st.write('Duck Duck Go Instant Answers: ', entity['duckIA'])

else:
    st.info("Choose at least one article from the dropbox on the sidebar \
        and press 'Execute pipeline' to start.")
