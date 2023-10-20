""" Module for parsing and string-formatting RTTM files.
"""

from typing import Tuple


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
    # line = line.decode("utf-8").strip()
    fields = line.split()
    if len(fields) < 9:
        raise IOError('Number of fields < 9. LINE: "%s"' % line)
    file_id = fields[1]
    speaker_id = fields[7]

    # Check valid turn onset.
    try:
        onset = float(fields[3])
    except ValueError as error:
        raise IOError('Segment onset not FLOAT. LINE: "%s"' % line) from error
    if onset < 0:
        raise IOError('Segment onset < 0 seconds. LINE: "%s"' % line)

    # Check valid turn duration.
    try:
        dur = float(fields[4])
    except ValueError as error:
        raise IOError('Segment duration not FLOAT. LINE: "%s"' % line) from error
    if dur <= 0:
        raise IOError('Segment duration <= 0 seconds. LINE: "%s"' % line)

    return (onset, dur, file_id, speaker_id)


# for segment in segments:


def format_rttm_line(segment: Tuple[float], file_id: str, speaker_id: str = "SPEECH"):
    """Formats a given segment of start and end times
    a line of RTTM file that is read in, ensures that the data
    is valid
    """
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
    # f.write(line.encode("utf-8"))
    # f.write(b"\n")

    return line
