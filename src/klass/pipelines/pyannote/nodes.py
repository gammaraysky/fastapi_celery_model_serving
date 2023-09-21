"""
This is a boilerplate pipeline 'pyannote'
generated using Kedro 0.18.11
"""
from typing import Dict, Callable

import soundfile as sf

def generate_uem_from_wav_files(wav_files: Dict[str, Callable]) -> str:
    """
    Concatenates information from WAV files to generate UEM (Unpartitioned
    Evaluation Map) formatted content.

    Args:
        wav_files (Dict[str, Callable]): A dictionary mapping the names of WAV
            files to their corresponding callables. The callables should return
            the duration of the WAV files as a numeric value.

    Returns:
        str: A string containing the UEM-formatted content with information
        about the WAV files. The format of each line is as follows:
        "<wav_file_name> NA 0.00 <duration>", where <wav_file_name> is the name
        of the WAV file, <duration> is the duration of the WAV file in seconds,
        and "NA" indicates that the file is unpartitioned.

    Example:
        wav_files = {
            "file1.wav": lambda: 10.2,
            "file2.wav": lambda: 5.8,
            "file3.wav": lambda: 7.3
        }
        uem_content = generate_uem_from_wav_files(wav_files)
        print(uem_content)
        # Output:
        # "file1.wav NA 0.00 10.2\nfile2.wav NA 0.00 5.8\nfile3.wav NA 0.00 7.3\n"
    """
    concatenated_content = ""
    for wav_file_name, wav_file_callable in wav_files.items():
        duration = wav_file_callable()
        concatenated_content += "{} NA 0.00 {}".format(wav_file_name, str(duration))
        concatenated_content += "\n"
    return concatenated_content

def generate_lst_from_rttm_files(rttm_files: Dict[str, Callable]) -> str:
    """Generates a concatenated string with RTTM file names separated by newlines.

    The function takes a dictionary of RTTM file Callables and generates a concatenated string
    containing the names of the RTTM files separated by newlines. The names are extracted from
    the keys of the input dictionary.

    Args:
        rttm_files (Dict[str, Callable]): A dictionary of RTTM file Callables. The keys represent
            the file names or identifiers, and the values are the Callables to read the content.

    Returns:
        str: A concatenated string with the names of the RTTM files separated by newlines.

    Example:
        >>> rttm_files = {
        ...     "file1.rttm": lambda: "SPKR-INFO ... <content of file1.rttm> ...",
        ...     "file2.rttm": lambda: "SPKR-INFO ... <content of file2.rttm> ...",
        ...     "file3.rttm": lambda: "SPKR-INFO ... <content of file3.rttm> ..."
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
    """Concatenates content from a dictionary of RTTM file Callables into a single RTTM string.

    The function takes a dictionary of RTTM file Callables and reads the content from each Callable.
    The content from all the Callables is concatenated and returned as a single RTTM string.

    Args:
        rttm_files (Dict[str, Callable]): A dictionary of RTTM file Callables. The keys represent
            the file names or identifiers, and the values are the Callables to read the content.

    Returns:
        str: The concatenated RTTM content as a single string.

    Raises:
        FileNotFoundError: If any of the input RTTM files are not found or if a Callable fails to read the file.
        IOError: If there is an error while reading the RTTM files.

    Note:
        This function is pure Python and has no side effects. It does not modify the input files.

    Example:
        >>> rttm_files = {
        ...     "file1.rttm": lambda: "SPKR-INFO ... <content of file1.rttm> ...",
        ...     "file2.rttm": lambda: "SPKR-INFO ... <content of file2.rttm> ...",
        ...     "file3.rttm": lambda: "SPKR-INFO ... <content of file3.rttm> ..."
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

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: {str(e)}")

    except IOError as e:
        raise IOError(f"Error: {str(e)}")

def concatenate_two_text_files(contents_file1: str, contents_file2: str) -> str:
    """Concatenate the contents of two text files.

    This function takes the contents of two text files
    as strings and concatenates them into a single string.

    Args:
        contents_file1 (str): Contents of the first text file as a string.
        contents_file2 (str): Contents of the second text file as a string.

    Returns:
        str: A string containing the concatenated contents of the two text files.
    """  
    try:
        concatenated_content = contents_file1 + contents_file2
        return concatenated_content
    except Exception as e:
        raise Exception(f"Error occurred while concatenating text files: {e}")
