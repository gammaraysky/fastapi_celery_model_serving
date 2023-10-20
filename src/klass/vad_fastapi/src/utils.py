"""
Web Frontend Utilities Module

This module provides utility functions for loading YAML configuration
files, validating file paths, sorting tasks by timestamp, parsing wave
file paths, and formatting task result presentations.

It also includes a class, `GetTaskDetails`, for retrieving task details
from an external API (Flower Dashboard) and performing data
transformations on those details.

Functions:
    - read_config(file_path): Load a YAML configuration file and return
      its contents as a dictionary.

    - is_valid_file_path(path): Check if a file path contains valid
      characters according to typical file systems.

    - custom_sort_key(item): Custom string sorting function for tasks
      based on their 'received' timestamp, handling None values.

    - parse_paths(input_wave_paths): Parse a string of wave file paths
      separated by various delimiters and return a list of cleaned and
      validated paths.



Class:
    - GetTaskDetails(base_mount_path, timezone): Initialize an instance
      of the `GetTaskDetails` class with a base mount path and timezone.

        - retrieve_one_task(task_id): Retrieve details for a single task
          using an external API and format the results.

        - retrieve_all_tasks(): Retrieve details for all tasks using an
        external API and format the results.

        - format_results_presentation(task_details, base_mount_path):
        Format task result presentations by converting full file paths
        to relative paths, formats and localises timestamps into string
        format and local timezone as specified in config YAML, and
        updating task states to include 'PARTIAL' or 'FAILURE' states.

        - remove_prefix_app_path(item_path, base_path_to_remove): Remove
        a prefix app path from file paths for simpler presentation.

        - fmt_datetime(timestamp, local_timezone): Format a floating-
        point timestamp into a string representing the corresponding
        local time, considering the specified timezone.

Examples:
    - See function docstrings for usage examples.


"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List

import pytz
import requests
import yaml
from fastapi import HTTPException
from requests.exceptions import JSONDecodeError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_config(file_path):
    """Load a YAML config file as dict"""
    with open(file_path, "r", encoding="utf8") as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    return config_data


def is_valid_file_path(path):
    """
    Check if a file path contains valid characters.

    This function verifies whether a given file path contains only
    valid characters as allowed by typical file systems.

    Args:
        `path` (str): The file path to be validated.

    Returns:
        `bool`: True if the file path contains valid characters, False
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


def custom_sort_key(item: str) -> str:
    """Custom string sorting function that handles None values.

    This function is used for sorting tasks based on their 'received'
    timestamp, allowing them to be presented in descending chronological
    order (latest first). However, in rare cases where the 'received'
    timestamp is not captured properly by the Celery broker, resulting
    in a None value, this function replaces it with an empty string to
    prevent sorting exceptions.

    Args:
        `item` (str): A timestamp in string format, or None.

    Returns:
        `str`: The timestamp in string format, or an empty string if it
            was None.
    """
    value = item[1]["received"] if item[1]["received"] is not None else ""
    return value


def parse_paths(input_wave_paths: str) -> List[str]:
    """Parses a string of wave file paths and returns a list of cleaned
    paths.

    This function takes an input string containing wave file paths
    separated by various delimiters (newline, double quotes, single
    quotes, or commas) and parses it into a list of cleaned paths. It
    removes leading and trailing whitespace, filters out empty paths,
    and ensures valid paths are returned.

    Args:
        `input_wave_paths` (str): A string containing wave file paths.

    Returns:
        `List[str]`: A list of cleaned and validated wave file paths.

    Example:
        >>> input_string = "a.wav\nb.wav\nc.wav"
        >>> parse_paths(input_string)
        ['a.wav', 'b.wav', 'c.wav']

        >>> input_string = "a.wav,b.wav, c.wav"
        >>> parse_paths(input_string)
        ['a.wav', 'b.wav', 'c.wav']

        >>> input_string = "'a.wav','b.wav','c.wav'"
        >>> parse_paths(input_string)
        ['a.wav', 'b.wav', 'c.wav']

        >>> input_string = 'single_path.wav'
        >>> parse_paths(input_string)
        ['single_path.wav']
    """
    # Check if the input string contains commas
    if "\n" in input_wave_paths:
        paths = input_wave_paths.split("\n")
    elif '"' in input_wave_paths:
        paths = input_wave_paths.split('"')
    elif "'" in input_wave_paths:
        paths = input_wave_paths.split("'")
    elif "," in input_wave_paths:
        paths = input_wave_paths.split(",")
    else:
        return [input_wave_paths.strip()]

    # Remove leading and trailing whitespace from each path
    paths = [path.strip() for path in paths]

    # Filter out any empty paths
    paths = [path for path in paths if len(path) > 0]

    # Filter out any malformed edge case splits (further testing needed)
    paths = [path for path in paths if path not in [".", ",", '"', "'"]]

    return paths


class GetTaskDetails:
    def __init__(self, base_mount_path: str, timezone: str):
        self.basepath = base_mount_path
        self.tz = timezone

    def retrieve_one_task(self, task_id: str) -> dict:
        """Retrieve details for a single task using an external API and
        format the results.

        Args:
            `task_id` (str): The unique identifier of the task to retrieve.

        Returns:
            `dict`: A dictionary containing formatted details of the retrieved
                task, including its state, timestamps, runtime, and more. If
                the task does not exist or an error occurs during retrieval,
                an empty dictionary is returned.

        Raises:
            HTTPException: If there is an issue with the API request, such
                as a JSON decoding error or an incorrect task ID. The
                exception includes a status code of 400 and a detail message
                describing the issue.

        Example:
            To retrieve details for a task with ID "task123", you can use:

            >>> task_details = get_task_details_instance.retrieve_one_task("task123")

            The returned dictionary may include the following keys:
            - `state`: The current state of the task.
            - `received`: The timestamp when the task was received.
            - `started`: The timestamp when the task started.
            - `succeeded`: The timestamp when the task succeeded.
            - `failed`: The timestamp when the task failed.
            - `rejected`: The timestamp when the task was rejected.
            - `retried`: The timestamp when the task was retried.
            - `retries`: The number of retries for the task.
            - `revoked`: The timestamp when the task was revoked.
            - `timestamp`: The timestamp associated with the task.
            - `runtime`: The runtime duration of the task.
            - `exception`: Information about any exception that occurred
            during the task.
            - `traceback`: The traceback information in case of an exception.
            - `args`: Arguments associated with the task (if available).
            - `result`: The result of the task (if available).
        """
        try:
            uri_info = "http://flower:5555/api/task/info/" + task_id
            resp_info = requests.get(uri_info, timeout=2)
            task = resp_info.json()
        except JSONDecodeError as exc_info:
            return None
            # raise HTTPException(
            #     status_code=400, detail="Please check the task ID is correct."
            # ) from exc_info

        fmt_task = {}
        fmt_task[task_id] = {
            "state": task["state"],
            "received": self.fmt_datetime(task["received"]),
            "started": self.fmt_datetime(task["started"]),
            "succeeded": self.fmt_datetime(task["succeeded"]),
            "failed": self.fmt_datetime(task["failed"]),
            "rejected": self.fmt_datetime(task["rejected"]),
            "retried": self.fmt_datetime(task["retried"]),
            "retries": task["retries"],
            "revoked": self.fmt_datetime(task["revoked"]),
            "timestamp": self.fmt_datetime(task["timestamp"]),
            "runtime": task["runtime"],
            "exception": task["exception"],
            "traceback": task["traceback"],
        }

        if task["args"] is not None:
            fmt_task[task_id]["args"] = eval(task["args"])[0]
        else:
            fmt_task[task_id]["args"] = task["args"]
        if task["result"] is not None:
            fmt_task[task_id]["result"] = eval(task["result"])
        else:
            fmt_task[task_id]["result"] = task["result"]

        fmt_task = self.format_results_presentation(fmt_task, self.basepath)

        return fmt_task

    def retrieve_all_tasks(self) -> dict:
        """Retrieve details for all tasks from an external API and
        format the results.

        This method queries an external API to retrieve details of all
        tasks and formats the results into a dictionary. The returned
        dictionary includes information about each task, such as its
        state, timestamps, runtime, exception details, and more.

        Returns:
            `dict`: A dictionary containing formatted details of all
                retrieved tasks. The dictionary is sorted in descending
                chronological order based on the 'received' timestamp of
                each task. If there are no tasks or an error occurs
                during retrieval, an empty dictionary is returned.

        Example:
            To retrieve details for all tasks, you can use:

            >>> all_task_details = get_task_details_instance.retrieve_all_tasks()

            The returned dictionary contains information about each task
            in the following format:
            - `state`: The current state of the task.
            - `received`: The timestamp when the task was received.
            - `started`: The timestamp when the task started.
            - `succeeded`: The timestamp when the task succeeded.
            - `failed`: The timestamp when the task failed.
            - `rejected`: The timestamp when the task was rejected.
            - `retried`: The timestamp when the task was retried.
            - `retries`: The number of retries for the task.
            - `revoked`: The timestamp when the task was revoked.
            - `timestamp`: The timestamp associated with the task.
            - `runtime`: The runtime duration of the task.
            - `exception`: Information about any exception that occurred
              during the task.
            - `traceback`: The traceback information in case of an
              exception.
            - `args`: Arguments associated with the task (if available).
            - `result`: The result of the task (if available).
        """
        # get all tasks from flower dashboard json api
        uri = "http://flower:5555/api/tasks"
        resp = requests.get(uri, timeout=4)
        tasks = resp.json()

        fmt_tasks = {}
        for task_id in list(tasks.keys()):
            fmt_tasks[task_id] = {
                "state": tasks[task_id]["state"],
                "received": self.fmt_datetime(tasks[task_id]["received"]),
                "started": self.fmt_datetime(tasks[task_id]["started"]),
                "succeeded": self.fmt_datetime(tasks[task_id]["succeeded"]),
                "failed": self.fmt_datetime(tasks[task_id]["failed"]),
                "rejected": self.fmt_datetime(tasks[task_id]["rejected"]),
                "retried": self.fmt_datetime(tasks[task_id]["retried"]),
                "retries": tasks[task_id]["retries"],
                "revoked": self.fmt_datetime(tasks[task_id]["revoked"]),
                "timestamp": self.fmt_datetime(tasks[task_id]["timestamp"]),
                "runtime": tasks[task_id]["runtime"],
                "exception": tasks[task_id]["exception"],
                "traceback": tasks[task_id]["traceback"],
            }

            if tasks[task_id]["args"] is not None:
                fmt_tasks[task_id]["args"] = eval(tasks[task_id]["args"])[0]
            else:
                fmt_tasks[task_id]["args"] = tasks[task_id]["args"]
            if tasks[task_id]["result"] is not None:
                fmt_tasks[task_id]["result"] = eval(tasks[task_id]["result"])
            else:
                fmt_tasks[task_id]["result"] = tasks[task_id]["result"]

        fmt_tasks = dict(
            sorted(
                fmt_tasks.items(),
                key=custom_sort_key,
                reverse=True,
            )
        )

        fmt_tasks = self.format_results_presentation(fmt_tasks, self.basepath)

        return fmt_tasks

    def format_results_presentation(
        self, task_details: dict, base_mount_path: str
    ) -> dict:
        """Formats task result presentation.

        This function performs two key tasks:

        1. Converts full file paths for audio and RTTM paths to relative
        file paths based on the volume mount point. In our FastAPI app,
        users provide inference files in a volume-mounted folder named
        'vol_mount,' which is mounted at '/usr/src/app/vol_mount.' While
        users can provide input files as relative paths within their
        'vol_mount' folder, inference uses full paths. This function
        transforms these full paths into user-friendly relative paths.

        2. Adjusts task states to 'PARTIAL' or 'FAILURE' for tasks
        containing errors. By default, Celery tasks report 'SUCCESS' if
        they return successfully and 'FAILURE' only if exceptions occur
        (and no results are returned). In our batch inference task, some
        files may succeed while others fail. This function updates
        states to:
        - 'FAILURE' if all files had errors.
        - 'PARTIAL' if some files had errors.

        Args:
            `task_details` (dict): Details of the requested task(s) by the
                client.

            `base_mount_path` (str): Prefix base path to remove from paths

        Returns:
            `dict`: Updated task details based on the criteria above.
        """
        task_det_upd = task_details

        # Update to relative filepaths
        for task_id in list(task_details.keys()):
            try:
                # update input paths (audio files only) if they exist
                for audio_file in list(
                    task_details[task_id]["args"]["audio_files"].keys()
                ):
                    audio_file_short = self.remove_prefix_app_path(
                        audio_file, base_mount_path
                    )
                    task_details[task_id]["args"]["audio_files"][
                        audio_file_short
                    ] = None
                    del task_details[task_id]["args"]["audio_files"][audio_file]
            except Exception as exc:
                logger.warning(
                    "Input args not found, ignoring - task %s - exception %s",
                    task_id,
                    exc,
                )
                # input audio files DO NOT exist
                pass

            try:
                # update output paths (audio files + rttm paths) if they
                # exist
                for audio_file in list(task_details[task_id]["result"].keys()):
                    audio_file_short = self.remove_prefix_app_path(
                        audio_file, base_mount_path
                    )

                    rttm_file = task_details[task_id]["result"][audio_file]["rttm_path"]
                    rttm_file_short = self.remove_prefix_app_path(
                        rttm_file, base_mount_path
                    )

                    task_det_upd[task_id]["result"][audio_file_short] = task_details[
                        task_id
                    ]["result"][audio_file]

                    task_det_upd[task_id]["result"][audio_file_short][
                        "rttm_path"
                    ] = rttm_file_short

                    del task_det_upd[task_id]["result"][audio_file]
            except Exception:
                # output results do not exist (could be still running, or
                # failed job)
                task_det_upd[task_id]["result"] = None

        # Update task states
        for task_id in list(task_details.keys()):
            # for every task, check how many of its input audio files
            # succeeded vs failed. if everything failed, set to "FAILURE".
            # if some failed, set to "PARTIAL"

            for task_id in list(task_details.keys()):
                if task_details[task_id]["result"] is None:
                    continue
                fail_count, success_count = 0, 0
                for audio_file in list(task_details[task_id]["result"].keys()):
                    if (
                        "SUCCESS"
                        in task_details[task_id]["result"][audio_file]["status"]
                    ):
                        success_count += 1
                    else:
                        fail_count += 1

                if fail_count > 0 and success_count == 0:
                    task_det_upd[task_id]["state"] = "FAILURE"
                elif fail_count > 0 and success_count > 0:
                    task_det_upd[task_id]["state"] = "PARTIAL"

        return task_det_upd

    def remove_prefix_app_path(self, item_path: str, base_path_to_remove: str) -> str:
        """Utility to remove prefix app path from inference audio/rttm
        files for simpler presentation.

        Args:
            `item_path` (str): original file path
            `base_path_to_remove` (str): prefix path to remove from item
                path

        Returns:
            `str`: file path with prefix trimmed.

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

    def fmt_datetime(self, timestamp: float) -> str:
        """
        Formats a floating-point timestamp into a string representing
        the corresponding local time, taking into account the specified
        timezone in the config yaml.

        Args:
            `timestamp` (float): A floating-point timestamp value.


        Returns:
            `str`: A string representing the UTC time in the format
                "YYYY-MM-DD HH:MM:SS". Returns None if there is an error
                during conversion.

        Examples:
            >>> fmt_datetime(1632569385.123456)
            '2021-09-25 15:23:05'

            >>> fmt_datetime(1632570000)
            '2021-09-25 15:33:20'

            >>> fmt_datetime("invalid_timestamp")
            None
        """
        try:
            # Format the datetime object as a string
            utc_datetime = datetime.utcfromtimestamp(timestamp)
            target_timezone = pytz.timezone(self.tz)
            localized_datetime = utc_datetime.astimezone(target_timezone)

            formatted_datetime = localized_datetime.strftime("%Y-%m-%d %H:%M:%S")

            return formatted_datetime
        except Exception:
            return None
