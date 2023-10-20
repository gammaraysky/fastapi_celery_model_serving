"""
This is a boilerplate pipeline 'pyannote_modeling'
generated using Kedro 0.18.11
"""
import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass.pipelines.pyannote_modeling.nodes import train_voice_activity_detection

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """
    This pipeline performs training of Pyannet VAD
    with logging of training loss over iterations and
    model callbacks

    Returns:
        kedro.pipeline.Pipeline: A kedro pipeline
                                 for model training
    """
    model_train_pipeline = pipeline(
        [
            node(
                func=train_voice_activity_detection,
                inputs=[
                    "params:seed",
                    "params:mlflow_tracking_uri",
                    "params:mlflow_experiment_name",
                    "params:database_config",
                    "params:vad_config",
                    "params:early_stopping_config",
                    "params:trainer_config",
                    "params:checkpoint_config",
                    "params:MLFLOW",
                    "combined_train_far.model_input_rttm_file",
                    "combined_val.model_input_rttm_file",
                    "combined_test.model_input_rttm_file",
                ],
                outputs="model_trained",
                name="train_voice_activity_detection",
                tags="baseline",
            ),
        ]
    )
    return model_train_pipeline
