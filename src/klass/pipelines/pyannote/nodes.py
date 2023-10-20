"""
This is a boilerplate pipeline 'pyannote'
generated using Kedro 0.18.11
"""
import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, List

import soundfile as sf

logger = logging.getLogger(__name__)


def generate_uem_from_wav_files(
    wav_file_infos: Dict[str, Callable], wav_files: Dict[str, Callable]
) -> List[str]:
    """
    Generates a Unified Evaluation Map (UEM) content based on provided
    WAV file information.

    Args:
        wav_file_infos (Dict[str, Callable]): A dictionary containing WAV
        file names as keys and callable functions that return soundfile
        information as values.

        wav_files (Dict[str, Callable]): A dictionary containing WAV
        file names as keys and callable functions that return audio file
        content as values. (Note that this input is unnecessary for the
        function. We include this so that Kedro's topological sort will
        note the prerequisite pipelines that need to run before this can
        run, because wav_files and wave_file_infos will point to the
        same folder)

    Returns:
        List[str]: A list of strings representing the UEM content, with
        each line containing the formatted information for a WAV file.

    Note:
        The UEM content format follows the pattern:
        "WAV_FILENAME NA 0.00 DURATION".
        Each line corresponds to a WAV file entry, where "WAV_FILENAME"
        is the name of the WAV file, "NA" is a placeholder, "0.00" is a
        placeholder, and "DURATION" is the duration of the WAV file.

    Example:
        wav_file_infos = {
            'file1.wav': get_soundfile_info1,
            'file2.wav': get_soundfile_info2,
        }
        uem_content = generate_uem_from_wav_file_infos(wav_file_infos)
        # Returns a list of strings containing UEM content for the
        # provided WAV files.
    """
    concatenated_content = ""
    for wav_file_name, soundfile_info_callable in wav_file_infos.items():
        soundfile_info = soundfile_info_callable()
        duration = soundfile_info.duration
        concatenated_content += "{} NA 0.00 {}".format(wav_file_name, str(duration))
        concatenated_content += "\n"
    return concatenated_content


def generate_lst_from_rttm_files(rttm_files: Dict[str, Callable]) -> str:
    """Generates a concatenated string with RTTM file names separated by
    newlines.

    The function takes a dictionary of RTTM file Callables and generates
    a concatenated string containing the names of the RTTM files
    separated by newlines. The names are extracted from the keys of the
    input dictionary.

    Args:
        rttm_files (Dict[str, Callable]): A dictionary of RTTM file
        Callables. The keys represent the file names or identifiers, and
        the values are the Callables to read the content.

    Returns:
        str: A concatenated string with the names of the RTTM files
        separated by newlines.

    Example:
        >>> rttm_files = {
        ...     "file1.rttm": lambda: "SPKR-INFO ...
        ...      <content of file1.rttm> ...",
        ...     "file2.rttm": lambda: "SPKR-INFO ...
        ...     <content of file2.rttm> ...",
        ...     "file3.rttm": lambda: "SPKR-INFO ...
        ...     <content of file3.rttm> ..."
        ... }
        >>> lst_content = generate_lst_from_rttm_files(rttm_files)
        >>> print(lst_content)
        'file1.rttm\nfile2.rttm\nfile3.rttm\n'
    """
    concatenated_content = ""
    for rttm_file_name in rttm_files.keys():
        concatenated_content += rttm_file_name
        concatenated_content += "\n"
    return concatenated_content


def concatenate_rttm_files(rttm_files: Dict[str, Callable]) -> str:
    """Concatenates content from a dictionary of RTTM file Callables
    into a single RTTM string.

    The function takes a dictionary of RTTM file Callables and reads the
    content from each Callable. The content from all the Callables is
    concatenated and returned as a single RTTM string.

    Args:
        rttm_files (Dict[str, Callable]): A dictionary of RTTM file
        Callables. The keys represent the file names or identifiers, and
        the values are the Callables to read the content.

    Returns:
        str: The concatenated RTTM content as a single string.

    Raises:
        FileNotFoundError: If any of the input RTTM files are not found
        or if a Callable fails to read the file.
        IOError: If there is an error while reading the RTTM files.

    Note:
        This function is pure Python and has no side effects.
        It does not modify the input files.

    Example:
        >>> rttm_files = {
        ...     "file1.rttm": lambda: "SPKR-INFO ...
        ...     <content of file1.rttm> ...",
        ...     "file2.rttm": lambda: "SPKR-INFO ...
        ...     <content of file2.rttm> ...",
        ...     "file3.rttm": lambda: "SPKR-INFO ...
        ...     <content of file3.rttm> ..."
        ... }
        >>> concatenated_rttm = concatenate_rttm_files(rttm_files)
        >>> print(concatenated_rttm)
        'SPKR-INFO ... <content of all RTTM files> ...'
    """
    try:
        concatenated_content = ""
        for rttm_file_callable in rttm_files.values():
            concatenated_content += rttm_file_callable()
        return concatenated_content

    except FileNotFoundError as error:
        raise FileNotFoundError(f"Error: {str(error)}")

    except IOError as error:
        raise IOError(f"Error: {str(error)}")


def concatenate_two_text_files(contents_file1: str, contents_file2: str) -> str:
    """Concatenate the contents of two text files.

    This function takes the contents of two text files
    as strings and concatenates them into a single string.

    Args:
        contents_file1 (str): Contents of the first text file as a
        string.
        contents_file2 (str): Contents of the second text file as a
        string.

    Returns:
        str: A string containing the concatenated contents of the two
        text files.
    """
    try:
        concatenated_content = contents_file1 + contents_file2
        return concatenated_content
    except Exception as error:
        raise Exception(f"Error occurred while concatenating text files: {error}")


def copy_wav_files(
    ali_far_train: Dict[str, Callable],
    ami_far_train: Dict[str, Callable],
    ali_far_val: Dict[str, Callable],
    ami_far_val: Dict[str, Callable],
    ali_far_test: Dict[str, Callable],
    ami_far_test: Dict[str, Callable],
) -> Dict[str, Callable]:
    """Copy wav files from original directory to target directories.

    This function maps files from original directory to target directories
    as strings, handling duplicates in the process.

    Args:
        ali_far_train (Dict[str, Callable]): original ali_far_train filepaths
        ami_far_train (Dict[str, Callable]): original ami_far_train filepaths
        ali_far_val (Dict[str, Callable]): original ali_far_val filepaths
        ami_far_val (Dict[str, Callable]): original ami_far_val filepaths
        ali_far_test (Dict[str, Callable]): original ali_far_test filepaths
        ami_far_test (Dict[str, Callable]): original ami_far_test filepaths

    Returns:
        Dict[str, Callable] : A collections of file names as key and the original
                              filepath of files
    """

    destination_directory = {}

    source_directories = [
        ali_far_train,
        ami_far_train,
        ali_far_val,
        ami_far_val,
        ali_far_test,
        ami_far_test,
    ]

    for source_directory in source_directories:
        for key in source_directory:
            if key in destination_directory:
                raise ValueError(f"Key '{key}' is repeated in input dictionaries.")
            destination_directory[key] = source_directory[key]

    return destination_directory
