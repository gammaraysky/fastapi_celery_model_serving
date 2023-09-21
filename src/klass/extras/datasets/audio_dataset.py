from pathlib import PurePosixPath
from typing import Tuple, Any, Dict
import fsspec 

import soundfile as sf
import numpy as np 

from kedro.io.core import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path

class AudioDataSet(AbstractDataSet[sf.SoundFile, Tuple[np.ndarray, int]]):
    """``AudioDataSet`` loads / saves audio data from a given filepath as a tuple containing a `numpy` array and an integer representing the sample rate using SoundFile.

    Example:
    ::

        >>> dataset = AudioDataSet(filepath='/audio/file/path.wav')

    Args:
        filepath (str): The location of the audio file to load / save data.
    """

    def __init__(self, filepath: str):
        """Creates a new instance of AudioDataSet to load / save audio data at the given filepath.

        Args:
            filepath (str): The location of the audio file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> Tuple[np.ndarray, int]:
        """Loads data from the audio file.

        Returns:
            Tuple[np.ndarray, int]: Data from the audio file as a tuple containing a numpy array and an integer representing the sample rate.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)
        
        # Using fsspec.open to open the audio file and obtain a file-like object
        with fsspec.open(self._filepath, 'rb') as f:
            audio_data, sample_rate = sf.read(f)
            return audio_data, sample_rate

    def _save(self, data: Tuple[np.ndarray, int]) -> None:
        """Saves audio data to the specified filepath.

        Args:
            data (Tuple[np.ndarray, int]: Audio data to be saved as a numpy array.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._filepath, self._protocol)

        # Unpack the data tuple into audio_data and sample_rate
        audio_data, sample_rate = data        

        # Using fsspec.open to open the audio file and obtain a file-like object
        with self._fs.open(save_path, "wb") as f:
            sf.write(save_path, audio_data, sample_rate)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing attributes of the dataset.
        """
        # Implementation for describing dataset attributes
        description = {
            'filepath': self._filepath,
            'dataset_type': 'audio',
            # Add more attributes if needed
        }
        return description

class SoundFileInfoDataSet(AbstractDataSet[sf.SoundFile, sf.info]):
    def __init__(self, filepath: str):
        """Creates a new instance of SoundFileInfoDataSet to load / save audio file information at the given filepath.

        Args:
            filepath (str): The location of the audio file to load / save audio file information.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> sf.info:
        """Loads data from the audio file.

        Returns:
            sf.info: An object with information about a SoundFile.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)
        
        # Using soundfile.info to get audio file information directly from the file path
        info = sf.info(load_path)
        return info

    def _save(self, info: sf.info) -> None:
        """Saves audio information to the specified filepath.

        Args:
            info (sf.info): SoundFile info object containing audio information.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._filepath, self._protocol)

        # Using soundfile.write to save the audio information to the file
        with open(save_path, "w") as f:
            f.write(f"Frames: {info.frames}\n")
            f.write(f"Samplerate: {info.samplerate}\n")
            f.write(f"Channels: {info.channels}\n")
            f.write(f"Subtype: {info.subtype}\n")

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing attributes of the dataset.
        """
        # Implementation for describing dataset attributes
        description = {
            'filepath': self._filepath,
            'dataset_type': 'audio',
            # Add more attributes if needed
        }
        return description

class DurationOnlyAudioDataSet(AudioDataSet):
    """Subclass of AudioDataSet that loads only the duration of the audio file."""

    def _load(self) -> Any:
        """Loads the duration of the audio file.

        Returns:
            Any: The duration of the audio file in seconds.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)

        # Load only the duration using sf.info()
        duration = sf.info(load_path).duration
        return duration

    def _save(self, data: Any) -> None:
        """Not implemented for DurationOnlyAudioDataSet as it only reads duration."""
        raise NotImplementedError("Saving is not supported in DurationOnlyAudioDataSet")

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing attributes of the dataset.
        """
        # Implementation for describing dataset attributes
        description = {
            'filepath': self._filepath,
            'dataset_type': 'audio',
            # Add more attributes if needed
        }
        return description