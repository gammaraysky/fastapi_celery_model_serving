"""
This is a boilerplate pipeline 'metric_evaluation'
generated using Kedro 0.18.12
"""
import json
import logging
import os
from pathlib import Path

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.database import FileFinder, registry

logger = logging.getLogger(__name__)

# To enable deterministic behaviour when using GPU, solves RunTimeError: To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def precision_score(tn: int, fp: int, fn: int, tp: int):
    """Calculates Precision score

    Args:
        tn (int): True negative
        fp (int): False positive
        fn (int): False negative
        tp (int): True positive
    Returns:
        float: Precision
    """
    return float(tp / (tp + fp))


def recall_score(tn: int, fp: int, fn: int, tp: int):
    """Calculates Recall score

    Args:
        tn (int): True negative
        fp (int): False positive
        fn (int): False negative
        tp (int): True positive
    Returns:
        float: Recall
    """
    return float(tp / (tp + fn))


def f1_score(tn: int, fp: int, fn: int, tp: int):
    """Calculates F1 score

    Args:
        tn (int): True negative
        fp (int): False positive
        fn (int): False negative
        tp (int): True positive
    Returns:
        float: F1
    """
    prec = precision_score(tn, fp, fn, tp)
    rec = recall_score(tn, fp, fn, tp)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def fmt(a, b, c):
    return (round(a, 4), round(b, 4), round(c, 4))


def positive_class(tn: int, fp: int, fn: int, tp: int):
    """Calculates the Precision, Recall and F1

    Args:
        tn (int): True negative
        fp (int): False positive
        fn (int): False negative
        tp (int): True positive
    Returns:
        tuple: Tuple of precision, recall and F1
    """
    prec = precision_score(tn, fp, fn, tp)
    rec = recall_score(tn, fp, fn, tp)
    f1 = f1_score(tn, fp, fn, tp)
    return fmt(prec, rec, f1)


def negative_class(tn: int, fp: int, fn: int, tp: int):
    """Calculates the Precision, Recall and F1 when the negative
    class is considered positive

    Args:
        tn (int): True negative
        fp (int): False positive
        fn (int): False negative
        tp (int): True positive
    Returns:
        tuple: Tuple of precision, recall and F1
    """
    prec_2 = precision_score(tp, fn, fp, tn)
    rec_2 = recall_score(tp, fn, fp, tn)
    f1_2 = f1_score(tp, fn, fp, tn)
    return fmt(prec_2, rec_2, f1_2)


def macro(tn: int, fp: int, fn: int, tp: int):
    """Calculates the macro-average of Precision, Recall and F1 for a 2-class scenario

    Args:
        tn (int): True negative (global)
        fp (int): False positive (global)
        fn (int): False negative (global)
        tp (int): True positive (global)
    Returns:
        tuple: Tuple of macro-precision, macro-recall and macro-F1
    """
    prec = precision_score(tn, fp, fn, tp)
    rec = recall_score(tn, fp, fn, tp)
    f1 = f1_score(tn, fp, fn, tp)

    prec_2 = precision_score(tp, fn, fp, tn)
    rec_2 = recall_score(tp, fn, fp, tn)
    f1_2 = f1_score(tp, fn, fp, tn)

    macro_prec = (prec + prec_2) / 2
    macro_rec = (rec + rec_2) / 2
    macro_f1 = (f1 + f1_2) / 2

    return fmt(macro_prec, macro_rec, macro_f1)


def micro(tn: int, fp: int, fn: int, tp: int):
    """Calculates the micro-average of Precision, Recall and F1 for a 2-class scenario

    Args:
        tn (int): True negative (global)
        fp (int): False positive (global)
        fn (int): False negative (global)
        tp (int): True positive (global)
    Returns:
        tuple: Tuple of micro-precision, micro-recall and micro-F1
    """
    prec = float((tp + tn) / (tp + tn + fp + fn))
    rec = float((tp + tn) / (tp + tn + fn + fp))
    f1 = 2 * (prec * rec) / (prec + rec)

    return fmt(prec, rec, f1)


def checkpoint_metric_evaluation(
    evaluation_database_config: str,
    pyannote_config: dict,
    checkpoint_base_path: str,
    pyannet_post_processing_config: dict,
    model_trained: dict,
) -> str:
    """Wrapper that ensures model_train is run by taking in arg
    model_trained json output from model_train pipeline.
    """
    return checkpoint_metric_evalonly(
        evaluation_database_config,
        pyannote_config,
        checkpoint_base_path,
        pyannet_post_processing_config,
    )


def checkpoint_metric_evalonly(
    evaluation_database_config: str,
    pyannote_config: dict,
    checkpoint_base_path: str,
    pyannet_post_processing_config: dict,
) -> str:
    """Metric evaluation for lightning checkpoints on ami-far and ali-far dataset

    Args:
        evaluation_database_config (str): Path to the database configuration file.
        pyannote_config (dict): Configuration for Pyannote protocol.
        checkpoint_base_path (str): Path to PyTorch Lightning checkpoints.
        pyannet_post_processing_config (dict): Configuration for Pyannet's model post-processing parameters.
        model_trained (dict): JSON file that exists if model train pipeline has been run beforehand. This is to force dependency to ensure evaluation can only run after training is completed.
    Returns:
        str : single JSON-formatted string reporting metrics generated
        for each lightning checkpoints

    Example of returned JSON:
    {"lightning_logs/version_0/checkpoints/epoch=0.ckpt\":
        {"Macro-Precision": 0.966,
        "Macro-Recall": 0.824,
        "Macro-F1": 0.901,
        "True Positive": 23173294 ,
        "True Negative": 291293,
        "False Positive": 123058,
        "False Negative": 123084,
        "duration": 2334.1
        }
    }
    """
    # ami-far and ali-far evaluation dataset as default
    registry.load_database(
        evaluation_database_config, mode="LoadingMode=LoadingMode.OVERRIDE"
    )

    # pyannote.database protocol
    protocol = registry.get_protocol(
        pyannote_config["protocol"], preprocessors={"audio": FileFinder()}
    )

    list_of_path = []
    dict_of_file_metrics = {}

    # Check that there are .ckpt files in the corresponding path given in metric_evaluation.yml
    if len(list(Path(checkpoint_base_path).rglob("*.ckpt"))) == 0:
        logger.error(
            "Metric evaluation on model checkpoint cannot be done as model checkpoint path/directory does not exist"
        )
        raise FileNotFoundError("Model Checkpoint path/directory does not exist")

    else:
        for path in Path(checkpoint_base_path).rglob("*.ckpt"):
            if path not in list_of_path:
                list_of_path.append(path)
                model = Model.from_pretrained(path)

                pipeline = VoiceActivityDetection(segmentation=model)

                HYPER_PARAMETERS = {
                    # onset/offset activation thresholds
                    "onset": pyannet_post_processing_config["onset"],
                    "offset": pyannet_post_processing_config["offset"],
                    # remove speech regions shorter than that many seconds.
                    "min_duration_on": pyannet_post_processing_config[
                        "min_duration_on"
                    ],
                    # fill non-speech regions shorter than that many seconds.
                    "min_duration_off": pyannet_post_processing_config[
                        "min_duration_off"
                    ],
                }

                pipeline.instantiate(HYPER_PARAMETERS)

                global_true_positive = 0
                global_false_positive = 0
                global_true_negative = 0
                global_false_negative = 0
                global_file_duration = 0

                for file in protocol.development():
                    true_positive = 0.0
                    true_negative = 0.0
                    false_positive = 0.0
                    false_negative = 0.0

                    # Generate the model predictions for speech regions
                    output = pipeline(file)
                    hypothesis = output.get_timeline().support()

                    # Get the ground truth .rttm stored in database
                    reference = file["annotation"].get_timeline().support()

                    # These are the non-speech regions taken from the ground truth .rttm
                    reference_ = reference.gaps(support=file["annotated"])
                    # These are the non-speech regions predicted by the model
                    hypothesis_ = hypothesis.gaps(support=file["annotated"])

                    # r and h are the time regions of speech from both the reference and hypothesis
                    for r, h in reference.co_iter(hypothesis):
                        true_positive += (r & h).duration
                    # r_ and h_ are the time regions of non-speech from both the reference_ and hypothesis_
                    for r_, h_ in reference_.co_iter(hypothesis_):
                        true_negative += (r_ & h_).duration

                    for r_, h in reference_.co_iter(hypothesis):
                        false_positive += (r_ & h).duration
                    for r, h_ in reference.co_iter(hypothesis_):
                        false_negative += (r & h_).duration

                    file_duration = reference.duration()

                    global_true_positive += true_positive
                    global_false_positive += false_positive
                    global_true_negative += true_negative
                    global_false_negative += false_negative
                    global_file_duration += file_duration

                macro_prec, macro_rec, macro_f1 = macro(
                    global_true_negative,
                    global_false_positive,
                    global_false_negative,
                    global_true_positive,
                )

                dict_of_file_metrics[f"{path}"] = {
                    "Macro-Precision": macro_prec,
                    "Macro-Recall": macro_rec,
                    "Macro-F1": macro_f1,
                    "True Positive": global_true_positive,
                    "True Negative": global_true_negative,
                    "False Positive": global_false_positive,
                    "False Negative": global_false_negative,
                    "duration": global_file_duration,
                }

                logger.info(f"Completed metric evaluation on checkpoint: {path}")

            json_content = json.dumps(
                dict_of_file_metrics,
                sort_keys=True,
            )

        return [json_content]
