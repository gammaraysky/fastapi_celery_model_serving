import logging
import os
import time
from pathlib import Path
from typing import List

import yaml
from celery import Celery, Task, states
from celery.exceptions import Ignore, Reject
from src.klass.pyannetmodel.model import PyanNetModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


########## LOAD FASTAPI YAML CONFIG ##########
def read_config(file_path):
    """
    Read a YAML configuration file and return its contents as a
    dictionary.

    Args:
        file_path (str): Path to the YAML configuration file to be read.

    Returns:
        dict: A dictionary containing the configuration data

    Example:
        >>> read_config("config.yml")
        {'key1': 'value1', 'key2': 'value2'}
    """
    with open(file_path, "r") as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    return config_data


cfg = read_config("conf/base/fastapi_deploy.yaml")
logger.info("Loaded YAML config.")

########## MODEL DEFINITIONS ##########
MODELS = cfg["models"]
MODEL_IDS = list(MODELS.keys())
logger.info("Loaded model definitions: %s", MODELS)

########## SETUP PATHS ##########
BASE_MOUNT_PATH = cfg["vol_mount_path"]
OUTPUT_RTTM_PATH = str(Path(BASE_MOUNT_PATH).joinpath(cfg["output_basepath"]))

########## INIT CELERY ##########
celery = Celery(__name__)
celery.conf.update(enable_utc=True, timezone="Asia/Singapore")
celery.conf.broker_url = os.environ.get(
    "CELERY_BROKER_URL", "amqp://admin:mypass@localhost:5672"
)
celery.conf.result_backend = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)


########## CELERY TASKS ##########
@celery.task(name="run_inference_pipeline", bind=True)  # throws=(ValueError, Ignore))
def run_inference_pipeline(self, wave_paths: List[str], model_id: str) -> dict:
    """
    Perform batch audio file inference using the specified model.

    This Celery task asynchronously performs batch inference on a list
    of audio file paths using the specified model identified by
    `model_id`. The results of the inference are processed and returned
    as a dictionary containing the output rttm paths.

    Args:
        self: A reference to the Celery task instance.
        wave_paths (List[str]): A list of file paths to audio files for
            batch inference.
        model_id (str): The identifier of the model to be used for
            inference.

    Returns:
        dict: A dictionary containing the output RTTM paths and status
            of the batch inference.

    """
    ### GET MODEL CHECKPOINT ###
    model_ckpt = MODELS[model_id]
    model_ckpt = str(Path(BASE_MOUNT_PATH).joinpath(model_ckpt))
    logger.info("Requested model: %s (%s)", model_id, model_ckpt)

    results = pyannet_model_predict(wave_paths, OUTPUT_RTTM_PATH, model_ckpt)
    # response["results"] = results

    ### THIS SECTION OMITTED.
    # Initial idea was to reflect a custom state e.g. 'INCOMPLETE'
    # for when a task is finished, but some files had errors
    # (e.g. misspelt or zero-byte input audio files)

    # However, in Celery's implementation, a custom state is only
    # temporary and will be overriden by Celery built-in states (i.e
    # SUCCESS) so long as you return from the task.

    # We need to return the task to propagate our partial results. If we
    # raise celery.exceptions.Ignore we can prevent celery from updating
    # the state to SUCCESS but results will not be returned, therefore
    # we won't have our rttm_paths for the files that did infer nicely.

    # Thus the choice is to simply return as SUCCESS, write our own
    # statuses for each individual audio file in a sub dictionary.

    ### OMITTED SECTION
    # error_found = False
    # for audio_file in results.keys():
    #     if "Error" in results[audio_file]["status"]:
    #         error_found = True
    #         logger.warning("Error found")
    #         break
    # if error_found:
    #     logger.warning("Update status failure")
    #     exc_info = {
    #         "exc_type": "Error",
    #         "exc_info": results,
    #     }
    #     self.update_state(state="INCOMPLETE", meta=exc_info)
    #     # we have to raise Ignore otherwise status will still reflect SUCCESS once this function exits successfully with no exceptions.
    #     raise Ignore()
    ##############

    return results


def pyannet_model_predict(
    wave_paths: List[str], output_path: str, model_ckpt: str
) -> dict:
    """
    Load a PyAnNet model and run predictions on a batch of audio files.

    This function loads a PyAnNet model checkpoint specified by
    `model_ckpt` and uses it to perform predictions on a batch of audio
    files specified by `wave_paths`. The predicted RTTM files are saved
    to the folder specified by `output_path`.

    Args:
        wave_paths (List[str]): List of file paths to audio files for
            inference.
        output_path (str): Folder path where predicted RTTM files will
            be saved.
        model_ckpt (str): Path to the PyAnNet model checkpoint.

    Returns:
        dict: A dictionary containing prediction results for each audio
            file in `wave_paths`. The keys are the input audio file
            paths, and the values are sub-dictionaries with the
            following keys:
        {
            'status' (str): The status of the prediction, e.g.
                'SUCCESS' or 'ERROR: File not found'.
            'rttm_path' (str): Path to the generated RTTM file.
        }

    Example:
        >>> predictions = pyannet_model_predict(
        ...     ["audio1.wav", "audio2.wav"],
        ...     "output/",
        ...     "model_checkpoint.ckpt"
        ... )
        >>> print(predictions)
        {
            'audio1.wav': {
                'status': 'SUCCESS',
                'rttm_path': 'output/audio1.rttm'
            },
            'audio2.wav': {
                'status': 'ERROR: Error during model inference',
                'rttm_path': None
            }
        }

    Notes:
        - This function uses the PyAnNet model to perform batch inference on audio files.
        - Predictions are saved as RTTM files in the specified output folder.
        - The 'status' key indicates whether the prediction was successful ('SUCCESS') or encountered
          an error ('ERROR') for each audio file.
        - Ensure that the model checkpoint and audio files exist at the specified paths.
    """
    pyannetmodel = PyanNetModel(model_ckpt)
    results = pyannetmodel.predict(wave_paths, output_path=output_path)
    return results
