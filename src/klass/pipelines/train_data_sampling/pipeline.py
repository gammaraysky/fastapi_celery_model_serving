"""
This module assembles a data processing pipeline that performs audio
sampling on the following datasets:
    - ali_far_train
    - ami_far_train
    - ali_near_train
    - ami_near_train
It utilizes the train_data_sampling function to sample audio data based
on specified configuration parameters.
"""
import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass.pipelines.train_data_sampling.nodes import train_data_sampling

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """Create a data processing pipeline for audio sampling on the
    following datasets:
    - ali_far_train
    - ami_far_train
    - ali_near_train
    - ami_near_train
    It utilizes the train_data_sampling function to sample audio data
    based on specified configuration parameters.

    Args:
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Pipeline: A data processing pipeline that samples audio data
            from different datasets and returns sampled wave and RTTM
            files.
    """
    # train sampling: normal (far field)
    train_sampling_pipeline = pipeline(
        [
            node(
                func=train_data_sampling,
                inputs=[
                    "chunked_wav_files",
                    "chunked_rttm_files",
                    "params:train_chunking_sampling_config",
                ],
                outputs=["sampled_wav_files", "sampled_rttm_files"],
            )
        ]
    )

    # train sampling: normal (far field)
    ali_far_train_sampling = pipeline(
        pipe=train_sampling_pipeline, namespace="ali_far_train", tags="baseline"
    )

    ami_far_train_sampling = pipeline(
        pipe=train_sampling_pipeline, namespace="ami_far_train", tags="baseline"
    )

    ali_near_train_sampling = pipeline(
        pipe=train_sampling_pipeline, namespace="ali_near_train"
    )

    ami_near_train_sampling = pipeline(
        pipe=train_sampling_pipeline, namespace="ami_near_train"
    )

    return (
        ali_far_train_sampling
        + ami_far_train_sampling
        + ali_near_train_sampling
        + ami_near_train_sampling
    )
    # return pipeline([])
