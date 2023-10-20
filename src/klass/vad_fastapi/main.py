"""
FastAPI Frontend Model Serving App

This FastAPI app provides a frontend interface for initiating batch
audio file inference tasks using specified models. It includes the
following routes:

1. `/` (Home):
   - Serves an HTML form for batch audio file inference.
   - Users can input a list of audio file paths and choose from a
     dropdown list of models.
   - Form submission triggers model inference on the provided audio
     files using the selected model.

2. `/inference` (Run Inference):
   - Allows users to submit a batch of audio file paths for model
     inference.
   - Input paths are sent to a Celery task for asynchronous batch
     processing.
   - Returns a unique task ID for tracking the task progress.

3. `/task/{task_id}` (Get Task):
   - Allows checking the status of a specific batch model inference
     task.
   - Provides details about the task's current state, timestamps, and
     results.

4. `/all_tasks` (Get All Tasks):
   - Allows checking the records of all model inference tasks initiated
     since inception.

5. `/health` (Health Check):
   - Performs health checks on various components of the system,
     including worker nodes, Redis backend, and RabbitMQ broker.
   - Returns an "OK" message if all components are operational.

Notes:
- The form submission on the Home route triggers the execution of batch
  model inference.
- Tasks are processed asynchronously using Celery workers.

"""

import logging
import subprocess
from pathlib import Path

import requests
from fastapi import FastAPI, Form, HTTPException, Request, status
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Json, ValidationError
from requests.auth import HTTPBasicAuth
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.klass.vad_fastapi.src.checkpoint_loader import CheckpointLoader
from src.klass.vad_fastapi.src.utils import (
    GetTaskDetails,
    is_valid_file_path,
    parse_paths,
    read_config,
)
from src.klass.vad_fastapi.worker import run_inference_pipeline

app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


########################################################################
# LOAD FASTAPI YAML CONFIG #############################################
cfg = read_config("conf/base/fastapi_deploy.yaml")
logger.info("Loaded YAML config.")

########################################################################
# MODEL DEFINITIONS ####################################################
MODEL_PATH = str(Path(cfg["vol_mount_path"]).joinpath(cfg["model_path"]))
MODELS = CheckpointLoader(MODEL_PATH).to_dict()

logger.info("Loaded model definitions: %s", MODELS)

########################################################################
# SETUP PATHS ##########################################################
BASE_MOUNT_PATH = cfg["vol_mount_path"]
OUTPUT_RTTM_PATH = str(Path(BASE_MOUNT_PATH).joinpath(cfg["output_basepath"]))
if Path(OUTPUT_RTTM_PATH).exists() is False:
    Path(OUTPUT_RTTM_PATH).mkdir(exist_ok=True, parents=True)
    logger.info("Created output rttm path: %s", OUTPUT_RTTM_PATH)

########################################################################
# INIT FASTAPI #########################################################
app = FastAPI()
app.mount(
    "/static", StaticFiles(directory="src/klass/vad_fastapi/static"), name="static"
)
templates = Jinja2Templates(directory="src/klass/vad_fastapi/templates")

########################################################################
# INIT TASK DETAIL RETRIEVER ###########################################
get_task_details = GetTaskDetails(BASE_MOUNT_PATH, cfg["timezone"])


########################################################################
# ROUTES ###############################################################
@app.get(
    "/",
    tags=["inference"],
    summary="Get inference request form",
)
def home(request: Request):
    """Serves an HTML form for batch audio file inference.

    This endpoint serves an HTML form where users can input a list of
    audio file paths, and choose from a dropdown list of specified
    models.
    Submitting the form will trigger the execution of model inference on
    the provided audio files using the selected model.

    The form includes the following fields:
    - `wave_paths` (Textarea): A textarea where users can enter a list
    of audio file paths, separated by newline characters.
    - `model_label` (Dropdown): A dropdown list containing predefined
    model options.

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
        context={"request": request, "model_labels": list(MODELS.keys())},
    )


@app.post(
    "/inference",
    tags=["inference"],
    summary="Run a batch inference task",
    status_code=status.HTTP_200_OK,
)
def run_inference(wave_paths: str = Form(...), model_label: str = Form(...)) -> dict:
    """Performs batch audio file inference using a Celery worker.

    This endpoint allows you to submit a batch of audio file paths for
    model inference using a Celery worker.
    The input file paths are sent to a Celery task for asynchronous
    batch processing, and the result can be retrieved once the task is
    completed.

    Args:
    - `wave_paths` (str): A list of file paths to audio files for
        batch inference, as a single string separated by newline or
        comma for each file path.

    - `model_label` (str):
        Example JSON request:
        ```
        {
            "wave_paths": "/path/to/audio1.wav\\n
                            /path/to/audio2.wav\\n
                            /path/to/audio3.wav",
            "model_label": "03697f1c-3cbd-4333-964b-6e7f935da45b-epoch=20"
        }
        ```

    Returns:
    - `str`: Unique task ID for tracking the task progress.

    Responses:
    - `200 OK`:
        Returns a JSON response with the task ID.
        Example JSON response:
        ```
        {
            "id": "68fba1de-cd9f-4025-8b26-fd8b60c50b1a"
        }
        ```
    - `422 Validation Error`:
        If the input data is invalid or missing.
        Example JSON response:
        ```
        {
            "detail": [
                {
                    "loc": [
                        "body",
                        "model_label"
                    ],
                    "msg": "field required",
                    "type": "value_error.missing"
                }
            ]
        }
        ```

    Examples:


        >>> curl -X POST "http://localhost:8004/inference" \\
        ...         -H 'Content-Type: multipart/form-data' \\
        ...         -F "model_label='03697f1c-3cbd-4333-964b-6e7f935da45b-epoch=20'" \\
        ...         -F "wave_paths='file 1.wav','file 2.wav'"
        {"id":"24f66132-5fc5-404e-8db8-89164c6bc619"}

        >>> curl -X POST "http://localhost:8004/inference" \\
        ...         -H 'Content-Type: multipart/form-data' \\
        ...         -F "model_label='03697f1c-3cbd-4333-964b-6e7f935da45b-epoch=20'" \\
        ...         -F "wave_paths=file1.wav,file2.wav"
        {"id":"4c7022fd-7acd-177b-6f8a-f7fcb0cac520"}
    """
    # split wave files into a list, and ensure valid filenames
    wave_paths = parse_paths(wave_paths)

    inference_args = {"audio_files": {}}
    # validate each wave path exists.
    # - if some exist, still create a job.
    # - if none exist, return HTTPException
    badfile_count = 0
    badfiles = []
    try:
        wave_paths = [
            str(Path(BASE_MOUNT_PATH).joinpath(w).resolve()) for w in wave_paths
        ]
    except Exception as exc_info:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "wave_paths"],
                    "msg": str(exc_info),
                    "type": type(exc_info),
                }
            ],
        ) from exc_info
    for wave_path in wave_paths:
        if Path(wave_path).is_file() is False:
            badfile_count += 1
            badfiles.append(wave_path)
        else:
            inference_args["audio_files"][wave_path] = None
    if badfile_count == len(wave_paths):
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "wave_paths"],
                    "msg": "All file paths could not be resolved",
                    "type": "FileNotFoundError",
                }
            ],
        )

    # validate model id and model checkpoint path exists
    try:
        model_ckpt = MODELS[model_label]["path"]
    except KeyError as exc_info:
        raise HTTPException(
            status_code=500,
            detail=[
                {
                    "loc": ["body", "model_label"],
                    "msg": f"Model UUID '{str(model_label)}' not found in config YAML.",
                    "type": str(type(exc_info)),
                }
            ],
        )
    if Path(model_ckpt).is_file() is False:
        raise HTTPException(
            status_code=500,
            detail=[
                {
                    "loc": ["body", "model_label"],
                    "msg": f"Model checkpoint {str(model_ckpt)} not found",
                    "type": "FileNotFoundError",
                }
            ],
        )

    inference_args["model_label"] = model_label

    task = run_inference_pipeline.delay(inference_args)

    if badfile_count > 0:
        return JSONResponse(
            {
                "id": task.id,
                "warning": "Some file paths could not be resolved. Task will proceed with remaining valid files.",
                "files_ok": list(inference_args["audio_files"].keys()),
                "files_affected": badfiles,
            }
        )
    return JSONResponse({"id": task.id})


@app.get(
    "/task/{task_id}",
    tags=["task status"],
    summary="Retrieve details on a given task",
)
def get_task(task_id: str) -> JSONResponse:
    """Checks the status of a batch model inference task.

    This endpoint allows users to check the status of a specific batch
    model inference task by providing its unique task identifier (`id`).

    Args:
    - `task_id` (str): The unique task identifier for the batch model
            inference task.

    Returns:
    - `JSONResponse`: Task details of requested task.

    JSON Response Example:
    ```
    {
        "1c0a1c2f-1546-49f5-8147-29d5193900b7": {
            "state": "PARTIAL",
            "received": "2023-09-25 16:26:14",
            "started": "2023-09-25 16:26:14",
            "succeeded": "2023-09-25 16:26:28",
            "failed": null,
            "rejected": null,
            "retried": null,
            "retries": 0,
            "revoked": null,
            "timestamp": "2023-09-25 16:26:28",
            "runtime": 14.051214977998825,
            "exception": null,
            "traceback": null,
            "args": {
                "model_label": "03697f1c-3cbd-4333-964b-6e7f935da45b-epoch=20",
                "audio_files": {
                    "b.wav": null,
                    "a.wav": null
                }
            },
            "result": {
                "b.wav": {
                    "status": "ERROR: File not found",
                    "rttm_path": null
                },
                "a.wav": {
                    "status": "SUCCESS",
                    "rttm_path": "output_rttm/a.rttm"
                }
            }
        }
    }

    Examples:

        >>> curl -X GET "http://localhost:8004/task/7e2b6785-77a4-493f-87b0-6bf277c0bf83"
        {"7e2b6785-77a4-493f-87b0-6bf277c0bf83":{"state": ... }}
    ```

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
    - `result` (dict) : An optional field containing the result of the
        completed task if available.
    """

    task_details = get_task_details.retrieve_one_task(task_id)
    if task_details is not None:
        return JSONResponse(task_details)
    else:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": f"{task_id}",
                    "msg": "Invalid task ID",
                    "type": "ValueError",
                }
            ],
        )


@app.get(
    "/all_tasks",
    tags=["task status"],
    summary="Retrieve details on all tasks",
)
def get_all_tasks() -> JSONResponse:
    """Checks the status of all tasks.

    This endpoint allows users to check the records of all model
    inference tasks initiated since inception.

    Returns:
    - `JSONResponse` : JSON response containing the statuses of all
        tasks.

    JSON Response Example:
    ```
    {
        "1c0a1c2f-1546-49f5-8147-29d5193900b7": {
            "state": "PARTIAL",
            "received": "2023-09-25 16:26:14",
            "started": "2023-09-25 16:26:14",
            "succeeded": "2023-09-25 16:26:28",
            "failed": null,
            "rejected": null,
            "retried": null,
            "retries": 0,
            "revoked": null,
            "timestamp": "2023-09-25 16:26:28",
            "runtime": 14.051214977998825,
            "exception": null,
            "traceback": null,
            "args": {
                "model_label": "03697f1c-3cbd-4333-964b-6e7f935da45b-epoch=20",
                "audio_files": {
                    "b.wav": null,
                    "a.wav": null
                }
            },
            "result": {
                "b.wav": {
                    "status": "ERROR: File not found",
                    "rttm_path": null
                },
                "a.wav": {
                    "status": "SUCCESS",
                    "rttm_path": "output_rttm/a.rttm"
                }
            }
        },
        "abcdefgh-ijkl-mnop-qrst-u0v0w0xyz002": { ... },
        "abcdefgh-ijkl-mnop-qrst-u0v0w0xyz003": { ... }
    }
    ```

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
    - `result` (dict) : An optional field containing the result of the
            completed task if available.
    """
    try:
        all_tasks_details = get_task_details.retrieve_all_tasks()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=exc) from exc
    return JSONResponse(all_tasks_details)


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def get_health() -> str:
    """Performs health checks on various components of the system.

    This endpoint performs health checks on the following components:
    1. Worker Nodes: It pings all worker nodes to ensure they are
    responsive and operational.
    2. Redis Backend Node: It pings the Redis backend node to verify its
    availability and responsiveness.
    3. RabbitMQ Broker Node: It pings the RabbitMQ broker node to
    confirm its availability and responsiveness.

    Returns:
    - `str`: "OK" if nominal operation, or specific description if any
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
    except subprocess.CalledProcessError as exc:
        logger.error("Pinging redis results backend failed with error: %s", exc)
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
    except Exception:
        logger.error("GET request to RabbitMQ vhosts API failed.")
        response_broker = False

    ### RETURN RELEVANT STATUS MESSAGE BASED ON FLAGS ###
    if response_workers and response_broker:  # response_redis
        return JSONResponse({"detail": "OK"})
    # elif not response_redis:
    #     raise HTTPException(
    #         status_code=500, detail="Redis node not responding")
    elif not response_workers:
        raise HTTPException(status_code=500, detail="Worker node(s) not responding")
    elif not response_broker:
        raise HTTPException(
            status_code=500, detail="RabbitMQ message broker node not responding"
        )
    else:
        raise HTTPException(
            status_code=500, detail="Unspecified error during healthcheck"
        )
