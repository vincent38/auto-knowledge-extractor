"""
Pipeline data_processing
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_data, scrap_data, save_datasets
from .nodes import print_all_texts, process_ner, do_coreference
from .nodes import process_sentiment_analysis, process_sense_disambiguation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                scrap_data,
                inputs=None,
                outputs=["rawArticles", "articlesDict"],
                name="scrap_data_node"
            ),
            node(
                load_data,
                inputs="rawArticles",
                outputs="articles",
                name="load_data_node"
            ),
            node(
                do_coreference,
                inputs=["articles", "articlesDict"],
                outputs=["corefArticles", "textArticlesDict"],
                name="do_coreference_node"
            ),
            node(
                print_all_texts,
                inputs="corefArticles",
                outputs=None,
                name="print_all_texts_node"
            ),
            node(
                process_ner,
                inputs="corefArticles",
                outputs="ner",
                name="process_ner_node"
            ),
            node(
                process_sense_disambiguation,
                inputs=["ner", "sentenceSentimentsSet"],
                outputs="entitiesDict",
                name="process_sense_disambiguation_node"
            ),
            node(
                process_sentiment_analysis,
                inputs=["corefArticles", "textArticlesDict"],
                outputs=["sentenceSentimentsSet", "sentArticlesDict"],
                name="process_sentiment_analysis_node"
            ),
            node(
                save_datasets,
                inputs=["sentArticlesDict", "entitiesDict"],
                outputs=["jsonArticlesDict", "jsonEntitiesDict"],
                name="save_datasets_node"
            )
        ]
    )