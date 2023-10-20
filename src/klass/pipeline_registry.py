"""Project pipelines."""
from __future__ import annotations

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import klass.pipelines.data_validation as dval
import klass.pipelines.metric_evaluation as me
import klass.pipelines.metric_eval_only as meo
import klass.pipelines.pyannote as p
import klass.pipelines.pyannote_modeling as pm
import klass.pipelines.train_data_augment as t_aug
import klass.pipelines.train_data_chunking as t_chunk
import klass.pipelines.train_data_sampling as t_samp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    # return pipelines

    validation = dval.create_pipeline()
    train_chunk = t_chunk.create_pipeline()
    train_samp = t_samp.create_pipeline()
    train_aug = t_aug.create_pipeline()
    pyannote_preprocessing_pipeline = p.create_pipeline()
    pyannote_modeling_pipeline = pm.create_pipeline()
    metric_evaluation_pipeline = me.create_pipeline()
    metric_evaluation_only = meo.create_pipeline()

    return {
        "__default__": validation
        + train_chunk
        + train_samp
        + train_aug
        + pyannote_preprocessing_pipeline
        + pyannote_modeling_pipeline
        + metric_evaluation_pipeline,
        "validation": validation,
        "train_chunk": train_chunk,
        "train_samp": train_samp,
        "train_aug": train_aug,
        "model_prepro": pyannote_preprocessing_pipeline,
        "model_train": pyannote_modeling_pipeline,
        "model_eval": metric_evaluation_only,
    }
