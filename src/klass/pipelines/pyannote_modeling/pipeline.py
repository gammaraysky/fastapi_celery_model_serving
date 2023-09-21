"""
This is a boilerplate pipeline 'pyannote_modeling'
generated using Kedro 0.18.11
"""
from klass.pipelines.pyannote_modeling.nodes import train_voice_activity_detection
from kedro.pipeline import Pipeline, node, pipeline

def create_pipeline(**kwargs) -> Pipeline:
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
                    ],
                    outputs=None,
                    name="train_voice_activity_detection"
            ),    
        ]
    )
    return model_train_pipeline
