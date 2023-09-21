"""
This is a boilerplate pipeline 'data_validation'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from klass.pipelines.data_validation.nodes import format_check

def create_pipeline(**kwargs) -> Pipeline:
    data_validation_pipeline = pipeline(
        [
            node(
                func=format_check,
                inputs=["soundfile_info"],
                outputs=["json_output"],
                name="format_check"
            )
        ]
    )
    return data_validation_pipeline