"""
This module assembles a data processing pipeline for audio chunking,
which is performed on the following datasets:
    - ali_far_train
    - ami_far_train
    - ali_near_train
    - ami_near_train
It utilizes the train_chunking function to chunk audio data based on
specified configuration parameters.
"""

import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass.pipelines.train_data_chunking.nodes import train_chunking

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """Create a data processing pipeline for audio chunking.

    This function assembles a data processing pipeline for audio
    chunking, which is performed on the following datasets:
    - ali_far_train
    - ami_far_train
    - ali_near_train
    - ami_near_train
    It utilizes the train_chunking function to chunk audio data based on
    specified configuration parameters.

    Args:
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Pipeline: A data processing pipeline that chunks audio data from
            different datasets and returns chunked wave and RTTM
            files.
    """
    train_chunking_pipeline = pipeline(
        [
            node(
                func=train_chunking,
                inputs=[
                    "primary_wav_files",
                    "primary_rttm_files",
                    "params:train_chunking_sampling_config",
                ],
                outputs=[
                    "chunked_wav_files",
                    "chunked_rttm_files",
                ],
                name="train_chunking_node",
            )
        ]
    )

    # ALI FAR
    ali_far_train_chunking = pipeline(
        pipe=train_chunking_pipeline, namespace="ali_far_train", tags="baseline"
    )

    # AMI FAR
    ami_far_train_chunking = pipeline(
        pipe=train_chunking_pipeline, namespace="ami_far_train", tags="baseline"
    )

    # ALI NEAR
    ali_near_train_chunking = pipeline(
        pipe=train_chunking_pipeline,
        namespace="ali_near_train",
    )

    # AMI NEAR
    ami_near_train_chunking = pipeline(
        pipe=train_chunking_pipeline,
        namespace="ami_near_train",
    )

    return (
        ali_far_train_chunking
        + ami_far_train_chunking
        + ali_near_train_chunking
        + ami_near_train_chunking
    )
    # return pipeline([])
