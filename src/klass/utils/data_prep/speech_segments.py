"""Speech segments module
Contains methods to manipulate speech annotations.
"""

import logging
import os
import xml.etree.ElementTree as eltree
from pathlib import Path
from typing import List, Tuple, Union  # Dict, Optional,

import numpy as np
from textgrid import TextGrid

logger = logging.getLogger(__name__)


def validate_segments(segments: List[Tuple]) -> List[Tuple[float]]:
    """Checks segments list for any invalid segments
    i.e segment start or end time is None, or if values are not floats

    Returns only a list of segments that are valid, simply dropping any
    invalid segments found.

    Args:
        segments (List[Tuple]): List of tuples of speech segment
            start and end times

    Returns:
        List[Tuple[float]]: List of segments with erroneous segments
            dropped.
    """
    validated = []
    for segment in segments:
        start, end = segment
        if start is not None and end is not None:
            if isinstance(start, float) and isinstance(end, float):
                if start >= 0 and end > start:
                    validated.append((start, end))

    difference = len(segments) - len(validated)
    if difference > 0:
        logger.warning(
            "Start or end segment missing, or duration 0. %s segments dropped",
            difference,
        )

    return validated


def merge_xmls_to_one_segment(
    xml_filelist: List[Union[Path, str]],
) -> List[Tuple[float]]:
    """Given a list of paths to several XML speech segments files, reads
    in and combines the speech segments, merging any overlaps, and
    outputs a combined speech segment.

    Args:
        xml_filelist (str): List of Paths to the XML segments files.

    Returns:
        list: List of tuples representing start and
            end times of each speech segment.

    Examples:
        >>> merge_xmls_to_one_segment([(5.1, 6.5), (6, 8), (9, 10)])
        [(5.1, 8), (9, 10)]
    """

    # read in multiple xml segments files
    segments_list = []
    for xml_filepath in xml_filelist:
        segments = read_xml(xml_filepath)
        segments_list.append(segments)

    # combine individual segments into one, merging overlaps
    segments_combined = merge_overlap_segments(segments_list)

    return segments_combined


def read_textgrid(textgrid_filepath: Union[str, Path]):
    """Extracts speech segments from a TextGrid file.

    Args:
        textgrid_filepath (str): Path to the TextGrid file.

    Returns:
        list: List of tuples representing start and end
                times of each speech segment.
    """
    # parse the textgrid file
    textgrid = TextGrid()
    if os.path.isfile(textgrid_filepath):
        textgrid.read(textgrid_filepath)

    speech_segments = []

    for tier in range(len(textgrid.tiers)):
        try:
            for interval in textgrid.tiers[tier]:
                start = interval.minTime
                end = interval.maxTime
                speech_segments.append((start, end))
        except Exception as e:
            logger.error(e)
            logger.error("Could not process: {}".format(textgrid_filepath))
            break

    return speech_segments


def read_rttm(rttm_filepath: Union[str, Path]):
    """Load speaker segments from RTTM file.

    For a description of the RTTM format, consult Appendix A of the
    NIST RT-09 evaluation plan.

    Args:
        rttmf (str) : Path to RTTM file.

    Returns:
        list: List of tuples representing start and end times of each
            speech segment.


    References:
        NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting
        Recognition Evaluation Plan.
        https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
    """
    with open(rttm_filepath, "rb") as f:
        segments = []
        # speaker_ids = set()
        # file_ids = set()
        for line in f:
            if line.startswith(b"SPKR-INFO"):
                continue
            start, dur, file_id, speaker_id = parse_rttm_line(line)

            segments.append((start, round(start + dur, 2)))
            # speaker_ids.add(turn.speaker_id)
            # file_ids.add(turn.file_id)

    return segments  # , speaker_ids, file_ids


def write_rttm(
    segments: List[Tuple[float]],
    output_rttm_filepath: Union[Path, str],
    file_id: str = None,
    speaker_id: str = "SPEECH",
) -> None:
    """Given a list of speech segments, write out to file in RTTM
    format.

    Args:
        segments (List[Tuple[float]]): List of speech segments in
            tuples of (start_time, end_time) in seconds.
        output_rttm_filepath (Path | str): Path to save RTTM file
    """
    """
        ### RTTM FILE FORMAT ###
        # Type -- segment type; should always by SPEAKER
        # File ID -- file name; basename of the recording minus
        #                       extension (e.g., rec1_a)
        # Channel ID -- channel (1-indexed) that turn is on; should
        #               always be 1
        # Turn Onset -- onset of turn in seconds from beginning of
        #               recording
        # Turn Duration -- duration of turn in seconds
        # Orthography Field -- should always by < NA >
        # Speaker Type -- should always be < NA >
        # Speaker Name -- name of speaker of turn; should be unique
        #                 within scope of each file
        # Confidence Score -- system confidence (probability) that
        #                     information is correct; should always be
        #                     < NA >
        # Signal Lookahead Time -- should always be < NA >
    """
    with open(output_rttm_filepath, "wb") as f:
        for segment in segments:
            start, end = segment
            duration = end - start
            fields = [
                "SPEAKER",
                file_id,
                "1",
                str(round(start, 2)),
                str(round(duration, 2)),
                "<NA>",
                "<NA>",
                speaker_id,
                "<NA>",
                "<NA>",
            ]
            line = " ".join(fields)
            f.write(line.encode("utf-8"))
            f.write(b"\n")


def read_xml(xml_filepath: Union[Path, str]) -> List[Tuple[float]]:
    """Reads in XML segments file as formatted in AMI dataset,
    whereby each speech segment is demarcated by a `segment` XML tag
    and contains attributes `transcriber_start` and
    `transcriber_end`.

    Args:
        xml_filepath (Path | str): path to XML segments files

    Returns:
        List[Tuple[float]]: A list of tuples of (start_time,
            end_time) for each speech segment
    """
    # Load the XML file
    tree = eltree.parse(xml_filepath)
    root = tree.getroot()

    # Iterate over segment tags
    segments = []
    for segment in root.iter("segment"):
        start = segment.get("transcriber_start")
        end = segment.get("transcriber_end")

        # if segment start and end is valid
        # (i.e start time smaller than end time)
        # add to our list, otherwise skip it
        if float(end) > float(start):
            segments.append((float(start), float(end)))

    return segments


def parse_rttm_line(line):
    """Parses a line of RTTM file that is read in, ensures that the data
    is valid

    Args:
        line (bytes): A line of an RTTM file, read in as binary

    Raises:
        IOError: If number of fields is less than 9
        IOError: If segment start is not float value
        IOError: If segment start is negative value
        IOError: If segment duration is not float
        IOError: If segment duration is 0 or negative

    Returns:
        Tuple: a tuple of (
                onset (float),
                dur (float),
                file_id (str),
                speaker_id (str)
            )

    """
    line = line.decode("utf-8").strip()
    fields = line.split()
    if len(fields) < 9:
        raise IOError('Number of fields < 9. LINE: "%s"' % line)
    file_id = fields[1]
    speaker_id = fields[7]

    # Check valid turn onset.
    try:
        onset = float(fields[3])
    except ValueError:
        raise IOError('Segment onset not FLOAT. LINE: "%s"' % line)
    if onset < 0:
        raise IOError('Segment onset < 0 seconds. LINE: "%s"' % line)

    # Check valid turn duration.
    try:
        dur = float(fields[4])
    except ValueError:
        raise IOError('Segment duration not FLOAT. LINE: "%s"' % line)
    if dur <= 0:
        raise IOError('Segment duration <= 0 seconds. LINE: "%s"' % line)

    return (onset, dur, file_id, speaker_id)


def merge_overlap_segments(
    segments_list: List[List[Tuple[float]]],
) -> List[Tuple[float]]:
    """Combines multiple speech segments lists into a single list.
    Where a single segment file may denote speech segments for a
    single speaker, and there are multiple speakers in a recording, this
    combines separate speech segments (that may have overlaps) and
    compiles into a single combined speech segment.

    Args:
        segments_list (List[List[Tuple[float]]]): A list of list of
            speech segments, where each segment is a tuple of start
            and end times in float seconds.

    Returns:
        List[Tuple[float]]: A combined list of speech segments,
            denoting only whenever there is speech across all the
            input speech segments given.

    Examples:
        >>> merge_overlap_segments([
                [(0, 1), (3, 5), (8, 9)],
                [(0.5, 1.5), (6, 7), (7.5, 8))]
            ])
        [[(0, 1.5), (3, 5), (6, 7), (7.5, 9)]]


    """
    # flatten list of speech segments
    segments_combined = []
    [segments_combined.extend(item) for item in segments_list]

    # sort the speech segments by ascending start time
    # (each segment is a tuple of (start, end) times
    segments_combined.sort(key=lambda x: x[0])

    # merge overlapping segments
    result = []
    for segment in segments_combined:
        # if this is the first segment
        # or if current segment start is larger than previous
        # segment end, then no overlap exists
        if not result or segment[0] > result[-1][1]:  # No overlap
            result.append(segment)
        # else: overlap exists
        # compare current segment end to previous segment end,
        # update previous segment end to whichever is larger.
        else:
            last_segment = result[-1]
            if segment[1] > last_segment[1]:
                result[-1] = (last_segment[0], segment[1])

    return result


def invert_segments(segments_sec: list, duration_secs: float) -> list:
    """Given a list of speech segments, and duration of signal,
    generates the inverse, i.e. nonspeech segments.

    Args:
        segments_sec (list of tuples): List of start and end times
            (in seconds) for samples containing speech.
        duration_sec (float): Duration of signal, in samples
        sr (int): Sampling rate. Will be used to convert back
            duration into seconds.

    Returns:
        list: A list of tuples of start and end samples of nonspeech
            segments

    Examples:
        >>> invert_segments(
                [(1.1, 2.2), (5.5, 6.6), (7.7, 8.8)],
                10)
        [(0, 1.1), (2.2, 5.5), (6.6, 7.7), (8.8, 10.0)]

    """

    # duration_secs = duration_samples / sr
    inverse_indices = []

    for i, (start, end) in enumerate(segments_sec):
        # if first speech segment and it does not start at sample 0
        # add in a nonspeech segment from sample 0 to current
        if i == 0 and start > 0:
            inverse_indices.append((0, start))

        # if last speech segment and it ends before last sample, add in
        # a nonspeech segment from current sample to the last sample
        if i == len(segments_sec) - 1 and end < duration_secs:
            inverse_indices.append((end, duration_secs))
            break

        # retrieve the next speech segment's start sample
        try:
            next_speech_start = segments_sec[i + 1][0]
            # and create a nonspeech segment starting from the current end
            # to next segment's start.
            inverse_indices.append((end, next_speech_start))
        except Exception:
            next_speech_start = None

    return inverse_indices


def convert_segments_seconds_to_samples(
    sr: int,
    segments: List,
) -> List:
    """Converts a list of segments denoted in seconds to a list of
    segments denoted in samples, based on sampling rate of given
    audio file.

    Args:
        sr (int): Audio sample rate in Hz e.g. 16000
        segments (List): List of segments denoted in seconds

    Returns:
        List: List of segments denoted in samples

    Examples:
        >>> sr = 2
        >>> segments = [(2.5, 10.0), (15.0, 36.0), (42.02, 60.77)]
        >>> anot = Annotations("data_annotations_path")
        >>> anot.convert_segments_seconds_to_sample(sr, segments)
        [(5, 20), (30, 72), (84, 122)]
    """

    segments_samples = []

    for starttime, endtime in segments:
        if isinstance(starttime, float) and isinstance(endtime, float):
            segments_samples.append(
                (int(round(starttime * sr)), int(round(endtime * sr)))
            )

    return segments_samples


def convert_segments_to_signal(
    duration_samples: int,
    sr: int,
    segments: List,
) -> np.array:
    """Converts a list of annotation segments into a signal array of
    0s and 1s, where 0 denotes absence and 1 denotes presence of a
    segment.

    Args:
        duration (int): Duration of signal, in samples
        sr (int): Audio sample rate in Hz e.g. 16000
        segments (List): List of segments denoted in seconds

    Returns:
        np.array: Annotated signal where 1 denotes segment is
            present.

    Examples:
        >>> duration_samples = 20
        >>> sample_rate = 2
        >>> merged_intervals = [(2.5, 4), (7.1, 9)]
        >>> anot = Annotations("data_annotations_path")
        >>> anot.convert_segments_to_signal(
                duration_samples,
                sample_rate,
                merged_intervals
            )
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    """

    # convert the segments denoted in seconds to denoted in samples first
    segments_samples = convert_segments_seconds_to_samples(sr, segments)

    # Initialize the signal array with zeros
    annotation_signal = np.zeros(duration_samples)

    # Set the corresponding segments in the signal array to 1
    for segment_start, segment_end in segments_samples:
        annotation_signal[segment_start:segment_end] = 1

    return annotation_signal


def concat_signal_segments(
    segments_sec: list,
    signal_sr_tuple: Tuple[np.ndarray, int],
) -> np.array:
    """Given an audio signal and a list of segments, return a
    concatenated signal of just the selected segments.

    Args:
        segments_sec (list): List of tuples containing selection
            start and end indices.
        signal (np.array): Original audio signal
        sr (int): Audio sample rate in Hz e.g. 16000

    Returns:
        np.array: Concatenated audio signal of just selected
            segments.

    Examples:
        >>> segments_sec = [(0, 2.5), (10.0, 15.0)]
        >>> signal_sr_tuple = [([
                55, 14, 72, 68, 99,
                77, 40, 45, 76, 6,
                23, 92, 66, 18, 61,
                89
            ])]
        >>> anot = Annotations("data_annotations_path")
        >>> anot.concat_signal_segments(segments_sec, signal_sr_tuple)
        [55, 14, 23, 92, 66, 18, 61]
    """
    signal, sr = signal_sr_tuple

    concat_signal = []
    for start, end in segments_sec:
        try:
            start_sample = int(round(start * sr))
            end_sample = int(round(end * sr))
            concat_signal.extend(signal[start_sample:end_sample])
        except Exception as e:
            continue

    return np.array(concat_signal)


def total_segment_duration(segments_sec: list) -> float:
    """Given a list of segments in seconds, collate the total duration."""
    durations = []
    for i, (start, end) in enumerate(segments_sec):
        if start > end:
            logger.warning("Line %s in segments: (%s, %s)", i, start, end)
            # raise ValueError("Found: start segment later than end segment")
            return None

        durations.append(end - start)

    return np.sum(durations)
