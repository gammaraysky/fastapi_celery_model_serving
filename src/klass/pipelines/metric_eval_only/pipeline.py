"""
This is a boilerplate pipeline 'metric_evaluation'
generated using Kedro 0.18.12
"""

import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass.pipelines.metric_evaluation.nodes import checkpoint_metric_evalonly

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    metric_evaluation_only = pipeline(
        [
            node(
                func=checkpoint_metric_evalonly,
                inputs=[
                    "params:evaluation_database_config",
                    "params:pyannote_config",
                    "params:checkpoint_base_path",
                    "params:pyannet_post_processing_config",
                ],
                outputs=[
                    "metric_report",
                ],
                name="checkpoint_metric_evalonly",
                tags="baseline",
            )
        ]
    )
    return metric_evaluation_only
