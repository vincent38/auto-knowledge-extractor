"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from ner_drinking_articles.pipelines import data_processing as dp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    dp_pipeline = dp.create_pipeline()
    return {
        "__default__": dp_pipeline,
        "dp": dp_pipeline
    }
