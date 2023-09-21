import os
from pathlib import Path


def check_path_exists(path: str):
    """
    Function checks if a path exists as a directory or file
    Args:
         path (str): directory path in string

    Returns:
         (Bool): True/False
    """
    if os.path.exists(path):
        return True

    else:
        return False


def create_directory(directory_path: str):
    """
    Function creates a directory if it does not exists
    Args:
         directory_path (str): directory path in string

    Returns:
         NA
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True)


def generate_rttm(
    list_of_start_times: list,
    list_of_duration: list,
    rttm_file_name: str,
    output_path: str,
):
    """
    Function generates an .rttm file in output_path folder, given a list of start times, duration and a rttm_file_name
    Args:
         list_of_start_times (list): List of start times denoting speech regions
         list_of_duration (list): List of durations denoting speech length
         rttm_file_name (str): Audio file name being processed
         output_path (str): Path to output generated .rttm file

    Returns:
         status (str): SUCCESS/ERROR
    """
    status = "SUCCESS"

    if len(list_of_start_times) != len(list_of_duration):
        status = "ERROR"
        return status, None

    if not check_path_exists(output_path):
        create_directory(output_path)

    output_rttm_path = "{}/{}.rttm".format(output_path, rttm_file_name)
    with open(output_rttm_path, "w") as file:
        for i in range(len(list_of_start_times)):
            file.write(
                "SPEAKER {} 1 {} {} <NA> <NA> SPEECH <NA> <NA>".format(
                    rttm_file_name, list_of_start_times[i], list_of_duration[i]
                )
            )
            file.write("\n")

    return status, output_rttm_path
