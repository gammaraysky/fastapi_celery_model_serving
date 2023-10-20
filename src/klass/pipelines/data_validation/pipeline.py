"""
This is a boilerplate pipeline 'data_validation'
generated using Kedro 0.18.12
"""

import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass.pipelines.data_validation.nodes import (
    validate_and_convert_wav_files,
)

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """Create a data processing pipeline for data validation on the
    following datasets:
    - ali_far
    - ami_far
    - ali_near
    - ami_near
    It utilizes the validate_and_convert_wav_files function to validate
    audio data based on specified configuration parameters.

    Args:
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Pipeline: A data processing pipeline that samples audio data
            from different datasets and returns sampled wave and RTTM
            files.
    """
    data_validation_pipeline = pipeline(
        [
            node(
                func=validate_and_convert_wav_files,
                inputs=[
                    "intermediate_wav_files",
                    "intermediate_wav_files_info",
                    "intermediate_rttm_files",
                ],
                outputs=[
                    "primary_rttm_files",
                    "primary_wav_files",
                    "format_check_report",
                ],
                name="validate_and_convert_wav_files",
            )
        ]
    )

    # ALI FAR
    ali_far_train_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ali_far_train", tags="baseline"
    )

    ali_far_val_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ali_far_val", tags="baseline"
    )

    ali_far_test_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ali_far_test", tags="baseline"
    )

    # AMI FAR
    ami_far_train_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ami_far_train", tags="baseline"
    )

    ami_far_val_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ami_far_val", tags="baseline"
    )

    ami_far_test_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ami_far_test", tags="baseline"
    )

    # ALI NEAR
    ali_near_train_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ali_near_train", tags="baseline"
    )

    ali_near_val_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ali_near_val", tags="baseline"
    )

    ali_near_test_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ali_near_test", tags="baseline"
    )

    # AMI NEAR
    ami_near_train_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ami_near_train", tags="baseline"
    )

    ami_near_val_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ami_near_val", tags="baseline"
    )

    ami_near_test_pipeline = pipeline(
        pipe=data_validation_pipeline, namespace="ami_near_test", tags="baseline"
    )

    return (
        ali_far_train_pipeline
        + ali_far_val_pipeline
        + ali_far_test_pipeline
        + ami_far_train_pipeline
        + ami_far_val_pipeline
        + ami_far_test_pipeline
        + ali_near_train_pipeline
        + ali_near_val_pipeline
        + ali_near_test_pipeline
        + ami_near_train_pipeline
        + ami_near_val_pipeline
        + ami_near_test_pipeline
    )
