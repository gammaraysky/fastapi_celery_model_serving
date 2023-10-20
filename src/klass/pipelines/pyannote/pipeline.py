"""
This is a boilerplate pipeline 'pyannote'
generated using Kedro 0.18.11
"""
import logging

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from klass.pipelines.pyannote.nodes import (
    concatenate_rttm_files,
    concatenate_two_text_files,
    generate_lst_from_rttm_files,
    generate_uem_from_wav_files,
)

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    """
    This pipeline performs handling of overlaps of speech
    across related rttms or within rttms
    and output a single unified rttm file.

    Returns:
        kedro.pipeline.Pipeline: A kedro pipeline
                                 for rttm reformating
    """
    pyannote_preprocess_pipeline = pipeline(
        [
            node(
                func=concatenate_rttm_files,
                inputs="primary_rttm_files",
                outputs="model_input_rttm_file",
                name="concatenate_rttm_files_node",
            ),
            node(
                func=generate_lst_from_rttm_files,
                inputs="primary_rttm_files",
                outputs="model_input_lst_file",
                name="generate_lst_from_rttm_files_node",
            ),
            node(
                func=generate_uem_from_wav_files,
                inputs=["primary_wav_files_info", "primary_wav_files"],
                outputs="model_input_uem_file",
                name="generate_uem_from_wav_files_node",
            ),
        ]
    )

    merge_two_files_pipeline = pipeline(
        [
            node(
                func=concatenate_two_text_files,
                inputs=["rttm_file_1", "rttm_file_2"],
                outputs="combined_rttm_file_output",
                name="concatenate_two_rttm_files_node",
            ),
            node(
                func=concatenate_two_text_files,
                inputs=["uem_file_1", "uem_file_2"],
                outputs="combined_uem_file_output",
                name="concatenate_two_uem_files_node",
            ),
            node(
                func=concatenate_two_text_files,
                inputs=["lst_file_1", "lst_file_2"],
                outputs="combined_lst_file_output",
                name="concatenate_two_lst_files_node",
            ),
        ]
    )

    ali_far_val_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline, namespace="ali_far_val", tags="baseline"
    )

    ami_far_val_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline, namespace="ami_far_val", tags="baseline"
    )

    ali_far_test_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline, namespace="ali_far_test", tags="baseline"
    )

    ami_far_test_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline, namespace="ami_far_test", tags="baseline"
    )

    ali_far_train_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        inputs={
            "primary_rttm_files": "ali_far_train.augmented_rttm_files",
            "primary_wav_files_info": "ali_far_train.augmented_wav_files_info",
            "primary_wav_files": "ali_far_train.augmented_wav_files",
        },
        namespace="ali_far_train",
        tags="baseline",
    )

    ali_near_train_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        inputs={
            "primary_rttm_files": "ali_near_train.augmented_rttm_files",
            "primary_wav_files_info": "ali_near_train.augmented_wav_files_info",
            "primary_wav_files": "ali_near_train.augmented_wav_files",
        },
        namespace="ali_near_train",
    )

    ami_far_train_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        inputs={
            "primary_rttm_files": "ami_far_train.augmented_rttm_files",
            "primary_wav_files_info": "ami_far_train.augmented_wav_files_info",
            "primary_wav_files": "ami_far_train.augmented_wav_files",
        },
        namespace="ami_far_train",
        tags="baseline",
    )

    ami_near_train_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        inputs={
            "primary_rttm_files": "ami_near_train.augmented_rttm_files",
            "primary_wav_files_info": "ami_near_train.augmented_wav_files_info",
            "primary_wav_files": "ami_near_train.augmented_wav_files",
        },
        namespace="ami_near_train",
    )

    combined_train_far_pipeline = pipeline(
        pipe=merge_two_files_pipeline,
        inputs={
            "rttm_file_1": "ali_far_train.model_input_rttm_file",
            "rttm_file_2": "ami_far_train.model_input_rttm_file",
            "uem_file_1": "ali_far_train.model_input_uem_file",
            "uem_file_2": "ami_far_train.model_input_uem_file",
            "lst_file_1": "ali_far_train.model_input_lst_file",
            "lst_file_2": "ami_far_train.model_input_lst_file",
        },
        outputs={
            "combined_rttm_file_output": "combined_train_far.model_input_rttm_file",
            "combined_uem_file_output": "combined_train_far.model_input_uem_file",
            "combined_lst_file_output": "combined_train_far.model_input_lst_file",
        },
        namespace="combined_train_far",
        tags="baseline",
    )

    combined_train_near_pipeline = pipeline(
        pipe=merge_two_files_pipeline,
        inputs={
            "rttm_file_1": "ali_near_train.model_input_rttm_file",
            "rttm_file_2": "ami_near_train.model_input_rttm_file",
            "uem_file_1": "ali_near_train.model_input_uem_file",
            "uem_file_2": "ami_near_train.model_input_uem_file",
            "lst_file_1": "ali_near_train.model_input_lst_file",
            "lst_file_2": "ami_near_train.model_input_lst_file",
        },
        outputs={
            "combined_rttm_file_output": "combined_train_near.model_input_rttm_file",
            "combined_uem_file_output": "combined_train_near.model_input_uem_file",
            "combined_lst_file_output": "combined_train_near.model_input_lst_file",
        },
        namespace="combined_train_near",
    )

    combined_train_pipeline = pipeline(
        pipe=merge_two_files_pipeline,
        inputs={
            "rttm_file_1": "combined_train_far.model_input_rttm_file",
            "rttm_file_2": "combined_train_near.model_input_rttm_file",
            "uem_file_1": "combined_train_far.model_input_uem_file",
            "uem_file_2": "combined_train_near.model_input_uem_file",
            "lst_file_1": "combined_train_far.model_input_lst_file",
            "lst_file_2": "combined_train_near.model_input_lst_file",
        },
        outputs={
            "combined_rttm_file_output": "combined_train.model_input_rttm_file",
            "combined_uem_file_output": "combined_train.model_input_uem_file",
            "combined_lst_file_output": "combined_train.model_input_lst_file",
        },
        namespace="combined_train",
    )

    combined_val_pipeline = pipeline(
        pipe=merge_two_files_pipeline,
        inputs={
            "rttm_file_1": "ali_far_val.model_input_rttm_file",
            "rttm_file_2": "ami_far_val.model_input_rttm_file",
            "uem_file_1": "ali_far_val.model_input_uem_file",
            "uem_file_2": "ami_far_val.model_input_uem_file",
            "lst_file_1": "ali_far_val.model_input_lst_file",
            "lst_file_2": "ami_far_val.model_input_lst_file",
        },
        outputs={
            "combined_rttm_file_output": "combined_val.model_input_rttm_file",
            "combined_uem_file_output": "combined_val.model_input_uem_file",
            "combined_lst_file_output": "combined_val.model_input_lst_file",
        },
        namespace="combined_val",
        tags="baseline",
    )

    combined_test_pipeline = pipeline(
        pipe=merge_two_files_pipeline,
        inputs={
            "rttm_file_1": "ali_far_test.model_input_rttm_file",
            "rttm_file_2": "ami_far_test.model_input_rttm_file",
            "uem_file_1": "ali_far_test.model_input_uem_file",
            "uem_file_2": "ami_far_test.model_input_uem_file",
            "lst_file_1": "ali_far_test.model_input_lst_file",
            "lst_file_2": "ami_far_test.model_input_lst_file",
        },
        outputs={
            "combined_rttm_file_output": "combined_test.model_input_rttm_file",
            "combined_uem_file_output": "combined_test.model_input_uem_file",
            "combined_lst_file_output": "combined_test.model_input_lst_file",
        },
        namespace="combined_test",
        tags="baseline",
    )

    return (
        ali_far_val_pipeline
        + ami_far_val_pipeline
        + ali_far_test_pipeline
        + ami_far_test_pipeline
        + ali_far_train_pipeline
        + ami_far_train_pipeline
        + ami_near_train_pipeline
        + ali_near_train_pipeline
        + combined_train_near_pipeline
        + combined_train_far_pipeline
        + combined_train_pipeline
        + combined_val_pipeline
        + combined_test_pipeline
    )
