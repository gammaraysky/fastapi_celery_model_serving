"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import klass.pipelines.pyannote as p
import klass.pipelines.pyannote_modeling as pm


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
#    pipelines = find_pipelines()
#    pipelines["__default__"] = sum(pipelines.values())
#    return pipelines
    pyannote_preprocessing_pipeline = p.create_pipeline()
    pyannote_modeling_pipeline = pm.create_pipeline()

    return {
        "__default__": pyannote_preprocessing_pipeline + pyannote_modeling_pipeline,
        "pyannote_preprocessing": pyannote_preprocessing_pipeline,
        "pyannote_modeling": pyannote_modeling_pipeline,        
    }
