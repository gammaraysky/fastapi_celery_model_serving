"""Load model checkpoints from a specified folder"""

import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CheckpointLoader:
    """Loads checkpoint .ckpt files from a specified directory and
    organizes them.

    This class provides functionality for loading checkpoint files
    (.ckpt) from a specified directory and organizing them into a
    dictionary for easy access.

    Args:
        path (str): The directory path containing the checkpoint files.

    Attributes:
        path (str): The directory path containing the checkpoint files.
        checkpoints (dict): A dictionary containing loaded checkpoint
            information.

    Methods:
        load_models_from_folder(): Load checkpoint models from the
            specified folder.
        to_dict() -> dict: Get the loaded checkpoints as a dictionary.

    Example:
        To use this class, instantiate it with the path to the
        checkpoint directory and then access the loaded checkpoints as a
        dictionary.

        ```python
        loader = CheckpointLoader("/path/to/checkpoints")
        checkpoints = loader.to_dict()
        print(checkpoints)
        ```
    """

    def __init__(self, path: str):
        """Initialize a CheckpointLoader instance.

        Args:
            path (str): The directory path containing the checkpoint
                files.
        """
        self.path = path
        self.checkpoints = self.load_models_from_folder()

    def load_models_from_folder(self):
        """Load checkpoint models from the specified folder and
        organize them.

        This method loads checkpoint models from the specified folder
        and organizes them into a dictionary for easy access.

        Returns:
            dict: A dictionary containing loaded checkpoint information.
        """
        checkpoints = {}
        ckpt_paths = list(Path(self.path).glob("*.ckpt"))

        for ckpt_path in ckpt_paths:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            model_label = str(checkpoint["uuid"]) + "-" + str(ckpt_path.stem)
            checkpoints[model_label] = {}
            checkpoints[model_label]["model_uuid"] = checkpoint["uuid"]
            checkpoints[model_label]["timestamp"] = checkpoint["timestamp"]
            checkpoints[model_label]["path"] = str(ckpt_path)
            del checkpoint

        sorted_checkpoints = dict(
            sorted(
                checkpoints.items(), key=lambda item: item[1]["timestamp"], reverse=True
            )
        )

        return sorted_checkpoints

    def to_dict(self) -> dict:
        """Get the loaded checkpoints as a dictionary.

        Returns:
            dict: A dictionary containing loaded checkpoint information.
        """
        return self.checkpoints
