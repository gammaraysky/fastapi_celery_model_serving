import pytest
from typing import Dict, Callable
from klass.pipelines.pyannote.nodes import (
    generate_uem_from_wav_files,
    generate_lst_from_rttm_files,
    concatenate_rttm_files,
    concatenate_two_text_files,
    copy_wav_files,
)


def test_generate_uem_content(mocker):
    # Mocking WAV file information
    mock_soundfile_info1 = mocker.MagicMock()
    mock_soundfile_info1.duration = 10.5
    mock_get_soundfile_info1 = mocker.MagicMock(return_value=mock_soundfile_info1)

    mock_soundfile_info2 = mocker.MagicMock()
    mock_soundfile_info2.duration = 20.25
    mock_get_soundfile_info2 = mocker.MagicMock(return_value=mock_soundfile_info2)

    wav_files = {
        "file1.wav": mock_get_soundfile_info1,
        "file2.wav": mock_get_soundfile_info2,
    }

    # Call the function
    uem_content = generate_uem_from_wav_files(wav_files)

    expected_content = "file1.wav NA 0.00 10.5\n" "file2.wav NA 0.00 20.25\n"
    assert uem_content == expected_content


def test_generate_lst_content():
    # Create mock RTTM files
    mock_rttm_files = {
        "file1.rttm": lambda: "...content of file1.rttm...",
        "file2.rttm": lambda: "...content of file2.rttm...",
        "file3.rttm": lambda: "...content of file3.rttm...",
    }

    # Test the function
    lst_content = generate_lst_from_rttm_files(mock_rttm_files)

    expected_content = "file1.rttm\n" "file2.rttm\n" "file3.rttm\n"

    assert lst_content == expected_content


def test_successful_concatenation():
    mock_rttm_files = {
        "file1.rttm": lambda: "content of file1.rttm",
        "file2.rttm": lambda: "content of file2.rttm",
        "file3.rttm": lambda: "content of file3.rttm",
    }

    expected_content = (
        "content of file1.rttm" "content of file2.rttm" "content of file3.rttm"
    )

    assert concatenate_rttm_files(mock_rttm_files) == expected_content


def test_file_not_found_error():
    def raise_file_not_found_error():
        raise FileNotFoundError("missing_file.rttm not found")

    mock_rttm_files = {
        "file1.rttm": lambda: "content of file1.rttm",
        "missing_file.rttm": raise_file_not_found_error,
        "file3.rttm": lambda: "content of file3.rttm",
    }

    with pytest.raises(FileNotFoundError, match="missing_file.rttm not found"):
        concatenate_rttm_files(mock_rttm_files)


def test_io_error():
    def raise_io_error():
        raise IOError("error while reading error_file.rttm")

    mock_rttm_files = {
        "file1.rttm": lambda: "content of file1.rttm",
        "error_file.rttm": raise_io_error,
        "file3.rttm": lambda: "content of file3.rttm",
    }

    with pytest.raises(IOError, match="error while reading error_file.rttm"):
        concatenate_rttm_files(mock_rttm_files)


def test_successful_concatenation():
    content1 = "This is the content of file1."
    content2 = "This is the content of file2."
    expected_result = content1 + content2

    assert concatenate_two_text_files(content1, content2) == expected_result


def test_concatenation_with_empty_string():
    content1 = "This is the content of file1."
    content2 = ""

    assert concatenate_two_text_files(content1, content2) == content1
    assert concatenate_two_text_files(content2, content1) == content1


def test_concatenation_of_both_empty_strings():
    content1 = ""
    content2 = ""

    assert concatenate_two_text_files(content1, content2) == ""


def test_concatenation_failure():  # Although rare, just to ensure our exception handler works
    with pytest.raises(
        Exception, match="Error occurred while concatenating text files"
    ):
        # Use a scenario that would cause an error. In our case, since string concatenation rarely fails,
        # we'll manually trigger the error for demonstration.
        faulty_content = {"not": "a string"}
        concatenate_two_text_files(faulty_content, "some content")


# This mock can be any callable. For simplicity, using lambda functions.
mock_callable1 = lambda: "content1"
mock_callable2 = lambda: "content2"
mock_callable3 = lambda: "content3"


def test_copy_wav_files_basic_merge():
    ali_far_train = {"file1": mock_callable1}
    ami_far_train = {"file2": mock_callable2}
    ali_far_val = {"file3": mock_callable3}
    ami_far_val = {}
    ali_far_test = {}
    ami_far_test = {}

    merged = copy_wav_files(
        ali_far_train,
        ami_far_train,
        ali_far_val,
        ami_far_val,
        ali_far_test,
        ami_far_test,
    )

    assert len(merged) == 3
    assert merged["file1"] == mock_callable1
    assert merged["file2"] == mock_callable2
    assert merged["file3"] == mock_callable3


def test_copy_wav_files_overlapping_keys_raises_error():
    ali_far_train = {"file1": mock_callable1}
    ami_far_train = {"file1": mock_callable2}

    with pytest.raises(ValueError) as e_info:
        copy_wav_files(ali_far_train, ami_far_train, {}, {}, {}, {})

    assert str(e_info.value) == "Key 'file1' is repeated in input dictionaries."


def test_copy_wav_files_empty_dictionaries():
    ali_far_train = {}
    ami_far_train = {"file1": mock_callable1}

    merged = copy_wav_files(ali_far_train, ami_far_train, {}, {}, {}, {})

    assert len(merged) == 1
    assert merged["file1"] == mock_callable1


def test_copy_wav_files_all_empty_dictionaries():
    merged = copy_wav_files({}, {}, {}, {}, {}, {})

    assert len(merged) == 0
