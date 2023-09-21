import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

# import redis
import requests
import yaml
from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from requests.auth import HTTPBasicAuth
from requests.compat import urljoin
from requests.exceptions import JSONDecodeError
from src.klass.vad_fastapi.worker import run_inference_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


########## LOAD FASTAPI YAML CONFIG ##########
def read_config(file_path):
    """Load a YAML config file as dict"""
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
if Path(OUTPUT_RTTM_PATH).exists() is False:
    Path(OUTPUT_RTTM_PATH).mkdir(exist_ok=True, parents=True)
    logger.info("Created output rttm path: %s", OUTPUT_RTTM_PATH)

########## INIT FASTAPI ##########
app = FastAPI()
app.mount(
    "/static", StaticFiles(directory="src/klass/vad_fastapi/static"), name="static"
)
templates = Jinja2Templates(directory="src/klass/vad_fastapi/templates")


########## ROUTES ##########
@app.get("/")
def home(request: Request):
    """Serve an HTML form for batch audio file inference using a
    specified model.

    This endpoint serves an HTML form where users can input a list of
    audio file paths, and choose from a dropdown list of specified
    models.
    Submitting the form will trigger the execution of model inference on
    the provided audio files using the selected model.

    The form includes the following fields:
    - `wave_paths` (Textarea): A textarea where users can enter a list
    of audio file paths, separated by newline characters.
    - `model_id` (Dropdown): A dropdown list containing predefined model
    options.

    The user-submitted form data will be sent as a POST request to the
    `/inference` endpoint for processing.

    Returns:
        HTML response containing the form for input.

    Notes:
        - The form submission will trigger the execution of batch model
          inference on the given audio files using the selected model.
          The POST `/inference` endpoint handles this processing logic.
    """

    return templates.TemplateResponse(
        "infer.html",
        context={"request": request, "model_ids": MODEL_IDS},
    )


@app.post("/inference", status_code=status.HTTP_200_OK)
def run_inference(
    wave_paths: str = Form(...),
    model_id: str = Form(...),
) -> dict:
    """Perform batch audio file inference using a Celery worker.

    This endpoint allows you to submit a batch of audio file paths for
    model inference using a Celery worker.
    The input file paths are sent to a Celery task for asynchronous
    batch processing, and the result can be retrieved once the task is
    completed.

    Args:
        wave_paths (str): A list of file paths to audio files for batch
            inference, as a single string separated by newline
            characters for each file path.
        model_id (str):
            Example JSON request:
            {
                "wave_paths": "/path/to/audio1.wav\n
                               /path/to/audio2.wav\n
                               /path/to/audio3.wav",
                "model_id": "model_v1.0"
            }

    Returns:
        str: The unique task identifier for tracking the Celery task's
            progress. You can use this ID to check the status or
            retrieve the result later.

    Responses:
        200 OK: Returns a JSON response with the task ID.
            Example JSON response:
            {
                "id": "abc123def456"
            }
        400 Bad Request: If the input data is invalid or missing.
            Example JSON response:
            {
                "detail": "Invalid paths provided."
            }

    Notes:
        - To check the status of a task or retrieve the result, you can
          make a GET request to the `/task/{task_id}` endpoint.
        - The Celery task `run_inference_pipeline` accepta a list of
          audio file paths as input and perform batch inference on them
          using your specified model.
        - Ensure that the Celery worker is running and configured
          properly for task execution.
        - This endpoint is suitable for batch audio inference tasks that
          require asynchronous processing.
    """

    # handle user input, split into multiple wave files and validate
    wave_paths = str(wave_paths).split("\n")
    wave_paths = [w.strip() for w in wave_paths]
    wave_paths = [w for w in wave_paths if len(w) > 0]
    try:
        wave_paths = [
            str(Path(BASE_MOUNT_PATH).joinpath(w).resolve()) for w in wave_paths
        ]
    except Exception as exc_info:
        # return JSONResponse({"Error": exc_info})
        raise HTTPException(status_code=400, detail=exc_info)

    for wave_path in wave_paths:
        if is_valid_file_path(wave_path) is False:
            raise HTTPException(status_code=400, detail="Invalid paths provided.")

    task = run_inference_pipeline.delay(wave_paths, model_id)
    return JSONResponse({"id": task.id})


@app.get("/task/{task_id}")
def get_task(task_id: str) -> JSONResponse:
    """Check the status of a batch model inference task.

    This endpoint allows users to check the status of a specific batch
    model inference task by providing its unique task identifier (`id`).

    Args:
        task_id (str): The unique task identifier for the batch model
            inference task.

    Returns:
        - JSON response containing the status of the task.

    JSON Response Example:
    {
        "c6de1c42-d416-45cd-9a4d-2ddaf10bd316":
        {
            "state": "SUCCESS",
            "received": "2023-09-21 03:54:09",
            "started": "2023-09-21 03:54:09",
            "succeeded": "2023-09-21 03:54:39",
            "failed": null,
            "rejected": null,
            "retried": null,
            "retries": 0,
            "revoked": null,
            "timestamp": "2023-09-21 03:54:39",
            "runtime": 30.258387592999497,
            "exception": null,
            "traceback": null,
            "args": [
                [
                    "file1.wav",
                    "file2.wav"
                ],
                "sincnet-v1"
            ],
            "result": {
                "file1.wav": {
                    "status": "ERROR: File not found",
                    "rttm_path": null
                },
                "file2.wav": {
                    "status": "SUCCESS",
                    "rttm_path": "output_rttm/file2.rttm"
                }
            }
        }
    }

    The JSON response includes the following fields:

    - `id` (str): The unique task identifier for the queried task.
    - `state` (str) : Current status of the task (e.g., "PENDING",
        "SUCCESS", "FAILURE").
    - `received` (str) : Timestamp task was received.
    - `started` (str) : Optional. Timestamp task started. None if not
            started.
    - `succeeded` (str) : Optional. Timestamp task completed with
            "SUCCESS" state. None if task has not completed as "SUCCESS"
    - `failed` (str) : Optional. Timestamp task completed with "FAILURE"
            state. None if task has not completed as "FAILURE"
    - `rejected` (str) : Optional. Timestamp task was rejected.
    - `retried` (str) : Optional. Timestamp task was retried.
    - `retries` (str) : Optional. Number of times task was retried.
    - `revoked` (str) : Optional. Timestamp task was revoked.
    - `timestamp` (str) : Optional. Timestamp task ended
    - `runtime` (str) : Optional. Duration of task run.
    - `exception` (str) : Optional. Exception details if an exception
            occurred
    - `traceback` (str) : Optional. Traceback details if an exception
            occurred
    - `args` (tuple) : The inputs provided for the inference task
    - `result` (dict) : An optional field containing the result of the completed task if available.

    Notes:
        - This endpoint is useful for monitoring the progress and
          completion of batch model inference tasks.
        - Ensure that the `id` provided is valid and corresponds to a
          previously initiated task.
    """

    try:
        uri_info = "http://flower:5555/api/task/info/" + task_id
        resp_info = requests.get(uri_info, timeout=2)
        task_info = resp_info.json()
    except JSONDecodeError as _:
        return "Please check the task ID is correct"

    task_details = {}
    task_details[task_id] = {
        "state": task_info["state"],
        "received": format_datetime(task_info["received"]),
        "started": format_datetime(task_info["started"]),
        "succeeded": format_datetime(task_info["succeeded"]),
        "failed": format_datetime(task_info["failed"]),
        "rejected": format_datetime(task_info["rejected"]),
        "retried": format_datetime(task_info["retried"]),
        "retries": task_info["retries"],
        "revoked": format_datetime(task_info["revoked"]),
        "timestamp": format_datetime(task_info["timestamp"]),
        "runtime": task_info["runtime"],
        "exception": task_info["exception"],
        "traceback": task_info["traceback"],
    }

    try:
        task_details[task_id]["args"] = eval(task_info["args"])
    except Exception as e:
        logger.info("args nope: %s", e)
        task_details[task_id]["args"] = task_info["args"]
    try:
        task_details[task_id]["result"] = eval(task_info["result"])
    except Exception as e:
        logger.info("results nope: %s", e)
        task_details[task_id]["result"] = task_info["result"]

    task_details = prettify_results(task_details)

    return JSONResponse(task_details)


@app.get("/all_tasks")
def get_all_tasks() -> JSONResponse:
    """Check the status of all tasks.

    This endpoint allows users to check the records of all model
    inference tasks initiated since inception.

    Returns:
        - JSON response containing the statuses of all tasks.

    JSON Response Example:
    {
        "abcdefgh-ijkl-mnop-qrst-u0v0w0xyz001":
        {
            "state": "SUCCESS",
            "received": "2023-09-21 03:54:09",
            "started": "2023-09-21 03:54:09",
            "succeeded": "2023-09-21 03:54:39",
            "failed": null,
            "rejected": null,
            "retried": null,
            "retries": 0,
            "revoked": null,
            "timestamp": "2023-09-21 03:54:39",
            "runtime": 30.258387592999497,
            "exception": null,
            "traceback": null,
            "args": [
                [
                    "file1.wav",
                    "file2.wav"
                ],
                "sincnet-v1"
            ],
            "result": {
                "file1.wav": {
                    "status": "ERROR: File not found",
                    "rttm_path": null
                },
                "file2.wav": {
                    "status": "SUCCESS",
                    "rttm_path": "output_rttm/file2.rttm"
                }
            }
        },
        "abcdefgh-ijkl-mnop-qrst-u0v0w0xyz002": { ... },
        "abcdefgh-ijkl-mnop-qrst-u0v0w0xyz003": { ... }

    }

    The JSON response includes the following fields:

    - `id` (str): The unique task identifier for the queried task.
    - `state` (str) : Current status of the task (e.g., "PENDING",
        "SUCCESS", "FAILURE").
    - `received` (str) : Timestamp task was received.
    - `started` (str) : Optional. Timestamp task started. None if not
            started.
    - `succeeded` (str) : Optional. Timestamp task completed with
            "SUCCESS" state. None if task has not completed as "SUCCESS"
    - `failed` (str) : Optional. Timestamp task completed with "FAILURE"
            state. None if task has not completed as "FAILURE"
    - `rejected` (str) : Optional. Timestamp task was rejected.
    - `retried` (str) : Optional. Timestamp task was retried.
    - `retries` (str) : Optional. Number of times task was retried.
    - `revoked` (str) : Optional. Timestamp task was revoked.
    - `timestamp` (str) : Optional. Timestamp task ended
    - `runtime` (str) : Optional. Duration of task run.
    - `exception` (str) : Optional. Exception details if an exception
            occurred
    - `traceback` (str) : Optional. Traceback details if an exception
            occurred
    - `args` (tuple) : The inputs provided for the inference task
    - `result` (dict) : An optional field containing the result of the completed task if available.

    Notes:
        - This endpoint is useful for monitoring the progress and
          completion of all batch model inference tasks.
    """
    # get all tasks from flower dashboard json api
    uri = "http://flower:5555/api/tasks"
    resp = requests.get(uri, timeout=4)
    alltasks = resp.json()

    formatted_alltasks = {}
    # logger.info("TASKS: %s", alltasks)
    for task_id in list(alltasks.keys()):
        formatted_alltasks[task_id] = {
            "state": alltasks[task_id]["state"],
            "received": format_datetime(alltasks[task_id]["received"]),
            "started": format_datetime(alltasks[task_id]["started"]),
            "succeeded": format_datetime(alltasks[task_id]["succeeded"]),
            "failed": format_datetime(alltasks[task_id]["failed"]),
            "rejected": format_datetime(alltasks[task_id]["rejected"]),
            "retried": format_datetime(alltasks[task_id]["retried"]),
            "retries": alltasks[task_id]["retries"],
            "revoked": format_datetime(alltasks[task_id]["revoked"]),
            "timestamp": format_datetime(alltasks[task_id]["timestamp"]),
            "runtime": alltasks[task_id]["runtime"],
            "exception": alltasks[task_id]["exception"],
            "traceback": alltasks[task_id]["traceback"],
        }

        try:
            formatted_alltasks[task_id]["args"] = eval(alltasks[task_id]["args"])
        except Exception as e:
            logger.info("args nope: %s", e)
            formatted_alltasks[task_id]["args"] = alltasks[task_id]["args"]
        try:
            formatted_alltasks[task_id]["result"] = eval(alltasks[task_id]["result"])
        except Exception as e:
            logger.info("results nope: %s", e)
            formatted_alltasks[task_id]["result"] = alltasks[task_id]["result"]

    formatted_alltasks = dict(
        sorted(
            formatted_alltasks.items(),
            key=lambda item: item[1]["received"],
            reverse=True,
        )
    )

    formatted_alltasks = prettify_results(formatted_alltasks)

    return JSONResponse(formatted_alltasks)


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def get_health() -> str:
    """Perform health checks on various components of the system.

    This endpoint performs health checks on the following components:
    1. Worker Nodes: It pings all worker nodes to ensure they are
    responsive and operational.
    2. Redis Backend Node: It pings the Redis backend node to verify its
    availability and responsiveness.
    3. RabbitMQ Broker Node: It pings the RabbitMQ broker node to
    confirm its availability and responsiveness.

    Returns:
        str: "OK" if nominal operation, or specific description if any
            nodes are unresponsive.
    """

    # ### CHECK REDIS RESULTS BACKEND NODE ###
    # response_redis = redis_db.ping()  # True if alive, False if dead

    ### CHECK CELERY WORKER NODE(S) ###
    command = "celery inspect ping -t 3"
    try:
        output = subprocess.check_output(command, shell=True, text=True)
        if "Error" in output:
            response_workers = False
        else:
            response_workers = True
    except subprocess.CalledProcessError as e:
        logger.error("Pinging redis results backend failed with error: %s", e)
        response_workers = False

    ### CHECK RABBITMQ BROKER NODE ###
    try:
        # essentially running:
        # curl -i -u admin:mypass http://rabbit:15672/api/vhosts
        uri = "http://rabbit:15672/api/vhosts"
        resp = requests.get(uri, auth=HTTPBasicAuth("admin", "mypass"), timeout=3)
        rabbitmq_state = resp.json()[0]["cluster_state"]["rabbit@rabbit"]
        if rabbitmq_state == "running":
            response_broker = True
        else:
            response_broker = False
    except Exception as e:
        logger.error("GET request to RabbitMQ vhosts API failed.")
        response_broker = False

    ### RETURN RELEVANT STATUS MESSAGE BASED ON FLAGS ###
    if response_workers and response_broker:  # response_redis
        return "OK"
    # elif not response_redis:
    #     return "Redis node not responding"
    elif not response_workers:
        return "Worker node(s) not responding"
    elif not response_broker:
        return "RabbitMQ message broker node not responding"
    else:
        return "Unspecified error during healthcheck"


########## HELPERS ##########
def prettify_results(task_details: dict) -> dict:
    """Converts full file paths for audio and RTTM paths to relative
    file paths based on the volume mount point.

    In the context of this FastAPI app, users provide inference files in
    a volume-mounted folder named 'vol_mount,' which is mounted at
    '/usr/src/app/vol_mount.' The client-side UI allows users to provide
    input files as relative paths within their host 'vol_mount' folder.
    However, during inference, the full path is used, resulting in both
    the audio input file and the output RTTM file having full paths.
    This function aims to transform these full paths back into relative
    paths, making it more user-friendly.

    Args:
        task_details (dict): Details of the requested task(s) by the
            client.

    Returns:
        dict: Task details with paths shortened to relative paths.
    """
    task_det_upd = task_details
    for task_id in list(task_details.keys()):
        for i, audio_file in enumerate(task_details[task_id]["args"][0]):
            audio_file_short = remove_prefix_app_path(audio_file, BASE_MOUNT_PATH)
            task_det_upd[task_id]["args"][0][i] = audio_file_short

        for audio_file in list(task_details[task_id]["result"].keys()):
            audio_file_short = remove_prefix_app_path(audio_file, BASE_MOUNT_PATH)

            rttm_file = task_details[task_id]["result"][audio_file]["rttm_path"]
            rttm_file_short = remove_prefix_app_path(rttm_file, BASE_MOUNT_PATH)

            task_det_upd[task_id]["result"][audio_file_short] = task_details[task_id][
                "result"
            ][audio_file]

            task_det_upd[task_id]["result"][audio_file_short][
                "rttm_path"
            ] = rttm_file_short

            del task_det_upd[task_id]["result"][audio_file]

    return task_det_upd


def remove_prefix_app_path(item_path: str, base_path_to_remove: str) -> str:
    """Utility to remove prefix app path from inference audio/rttm files
    for simpler presentation.

    Args:
        item_path (str): original file path
        base_path_to_remove (str): prefix path to remove from item path

    Returns:
        str: file path with prefix trimmed.

    Examples:
        >>> remove_prefix_app_path(
                "/usr/src/app/vol_mount/output_rttm/file1.rttm",
                "/usr/src/app/vol_mount")
        "output_rttm/file1.rttm"
    """
    if item_path is None:
        return item_path

    app_path = str(Path(base_path_to_remove))
    if item_path.startswith(app_path):
        item_path = item_path[len(app_path) :]
    if item_path.startswith("/"):
        item_path = item_path[1:]

    return item_path


def format_datetime(timestamp: float) -> str:
    """
    Formats a floating-point timestamp into a string representing the
    corresponding UTC time.

    Args:
        timestamp (float): A floating-point timestamp value.

    Returns:
        str: A string representing the UTC time in the format
            "YYYY-MM-DD HH:MM:SS". Returns None if there is an error
            during conversion.
    """
    try:
        # Convert the timestamp to a datetime object
        datetime_obj = datetime.fromtimestamp(timestamp)

        # Format the datetime object as a string
        formatted_datetime = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_datetime
    except:
        return None


def is_valid_file_path(path):
    """
    Check if a file path contains valid characters.

    This function verifies whether a given file path contains only
    valid characters as allowed by typical file systems.

    Args:
        path (str): The file path to be validated.

    Returns:
        bool: True if the file path contains valid characters, False
            otherwise.

    Examples:
        >>> is_valid_file_path("/path/to/file.txt")
        True
        >>> is_valid_file_path("C:\\My Documents\\file.txt")
        False
        >>> is_valid_file_path("/var/data/file with spaces.txt")
        False
    """
    # Check for valid characters
    valid_chars = re.compile(r"^[\w\-./]+$")
    return valid_chars.match(path) is not None


################################################################
########################## DEPRECATED ##########################
################################################################

########## INIT REDIS BACKEND DB STORE ##########
# redis_db = redis.Redis(host="redis", port=6379, decode_responses=True)

## this previous method below was reading in using redis backend.
## replaced, reading directly from flower json api,
## since flower dashboard data is now persistent.

# @app.get("/ended")
# def get_endedjobs() -> dict:
#     """Ended endpoint. Retrieves all tasks stored in results backend
#     database, i.e. tasks that have concluded, and have status either
#     "FAILURE" or "SUCCESS". ("PENDING" will not be listed here.)

#     Returns:
#         dict: Task details of tasks that have concluded
#     """
#     db_entries = list(redis_db.keys())

#     # filter only tasks
#     db_entries = [entry for entry in db_entries if entry.startswith("celery")]

#     results_dict = {}

#     if len(db_entries) <= 0:
#         return results_dict

#     # LOOP THRU ALL FINISHED TASKS
#     for db_entry in db_entries:
#         entry = json.loads(redis_db.get(db_entry))

#         # FOR EACH TASK, REFORMAT THE OUTPUT DICT:
#         # - REMOVE FILE PATH PREFIXES
#         # - CHECK AND UPDATE TASK STATUS TO FAILURE
#         #   IF ANY AUDIO FILES HAD ERRORS
#         error_found = False
#         entry_updated = {}
#         # entry_updated["traceback"] = entry["traceback"]
#         # entry_updated["children"] = entry["children"]
#         entry_updated["date_done"] = entry["date_done"]
#         entry_updated["result"] = {}

#         for audio_file in entry["result"].keys():
#             audio_shortened = remove_prefix_app_path(audio_file, BASE_MOUNT_PATH)

#             rttm_shortened = remove_prefix_app_path(
#                 entry["result"][audio_file]["rttm_path"], BASE_MOUNT_PATH
#             )

#             entry_updated["result"][audio_shortened] = entry["result"][audio_file]

#             entry_updated["result"][audio_shortened]["rttm_path"] = rttm_shortened

#             if "ERROR" in entry_updated["result"][audio_shortened]["status"]:
#                 error_found = True

#         if error_found:
#             entry_updated["status"] = "ERRORS"
#         else:
#             entry_updated["status"] = entry["status"]

#         results_dict[entry["task_id"]] = {
#             "status": entry_updated["status"],
#             "result": entry_updated["result"],
#             # "traceback": entry_updated["traceback"],
#             # "children": entry_updated["children"],
#             "date_done": entry_updated["date_done"],
#         }

#     results_sorted_date_done = dict(
#         sorted(
#             results_dict.items(), key=lambda item: item[1]["date_done"], reverse=True
#         )
#     )

#     return results_sorted_date_done


## this previous method below was reading in using celery.result.AsyncResult.
## replaced, reading directly from flower json api,
## since flower dashboard data is now persistent.

# @app.get("/status/{task_id}")
# def get_status(task_id: str):
#     """Status endpoint. Checks task status/results for a given task id.

#     Args:
#         task_id (str): alphanumeric string of task ID.

#     Returns:
#         dict: Task details of requested task
#     """
#     task_result = AsyncResult(task_id)
#     # logger.info("TS: %s", task_result.status)

#     try:
#         # if job is pending, task_result.result is Nonetype, handle it
#         if type(task_result.result) != dict:
#             return {
#                 task_id: {
#                     "status": task_result.status,
#                     "result": task_result.result,
#                     # "traceback": task_result.traceback,
#                     # "children": task_result.children,
#                     "date_done": task_result.date_done,
#                 }
#             }

#         # if job is done, format result audio input and rttm output paths
#         # without the app path prefixes, and update task status to failure
#         # if any audio files had errors during inference.
#         else:
#             results = task_result.result
#             results_updated = {}

#             error_found = False
#             for audio_file in list(results.keys()):
#                 audio_shortened = remove_prefix_app_path(audio_file, BASE_MOUNT_PATH)

#                 rttm_shortened = remove_prefix_app_path(
#                     results[audio_file]["rttm_path"], BASE_MOUNT_PATH
#                 )

#                 results_updated[audio_shortened] = results[audio_file]

#                 results_updated[audio_shortened]["rttm_path"] = rttm_shortened

#                 if "ERROR" in results_updated[audio_shortened]["status"]:
#                     error_found = True

#             if error_found:
#                 task_status = "ERRORS"
#             else:
#                 task_status = task_result.status

#             return {
#                 task_id: {
#                     "status": task_status,
#                     "result": results_updated,
#                     # "traceback": task_result.traceback,
#                     # "children": task_result.children,
#                     "date_done": task_result.date_done,
#                 }
#             }

#     except Exception as e:
#         # handles out of scope cases, where task_result.result is ready
#         # but does not contain result. Occurred when we used custom
#         # state INCOMPLETE but this has been deprecated.
#         logger.error(e)
#         return e
