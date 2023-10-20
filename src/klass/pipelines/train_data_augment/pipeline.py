"""
Create a data processing pipeline for audio data augmentation.

This module assembles a data processing pipeline for audio data
augmentation. It performs augmentation  on the following datasets:
    - ali_far_train
    - ami_far_train
    - ali_near_train
    - ami_near_train
utilizing the train_data_augment function. The pipeline takes in chunked
audio and RTTM data, as well as configuration parameters for chunking
and augmentation.

"""
import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass.pipelines.train_data_augment.nodes import train_data_augment

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """Create a data processing pipeline for audio data augmentation.

    This function assembles a data processing pipeline for audio data
    augmentation. It performs augmentation on the following datasets:
    - ali_far_train
    - ami_far_train
    - ali_near_train
    - ami_near_train
    utilizing the train_data_augment function. The pipeline takes in
    chunked audio and RTTM data, as well as configuration parameters for
    chunking and augmentation.

    Args:
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Pipeline: A data processing pipeline that augments audio data
            from different datasets and returns augmented wave files,
            augmented RTTM files, and an augmentation report in JSON.
    """
    train_augment_pipeline = pipeline(
        [
            node(
                func=train_data_augment,
                inputs=[
                    "sampled_wav_files",
                    "sampled_rttm_files",
                    "params:train_chunking_sampling_config",
                    "params:train_augment_config",
                ],
                outputs=[
                    "augmented_wav_files",
                    "augmented_rttm_files",
                    "augmented_report",
                ],
            )
        ]
    )

    # train augment: normal (far field)
    ali_far_train_augment = pipeline(
        pipe=train_augment_pipeline, namespace="ali_far_train", tags="baseline"
    )

    ami_far_train_augment = pipeline(
        pipe=train_augment_pipeline, namespace="ami_far_train", tags="baseline"
    )

    # train augment: augmented (near field)
    ali_near_train_augment = pipeline(
        pipe=train_augment_pipeline,
        namespace="ali_near_train",
    )

    ami_near_train_augment = pipeline(
        pipe=train_augment_pipeline,
        namespace="ami_near_train",
    )

    return (
        ali_far_train_augment
        + ami_far_train_augment
        + ali_near_train_augment
        + ami_near_train_augment
    )
