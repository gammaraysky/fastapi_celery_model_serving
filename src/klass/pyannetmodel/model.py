"""
model_params for PyanNetModel are of the form:
{
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
"""
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
import logging
from pyannote.audio import Model as md
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.database import FileFinder, registry

from src.klass.pyannetmodel import generate_inference_rttm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(ABC):
    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike],
        model_params: Optional[dict] = None,
        finetune_params: Optional[dict] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_params = model_params
        self.finetune_params = finetune_params
        self.model = None
        self.model_summary = None

    @abstractmethod
    def predict(self, inference_audio_paths: list):
        pass

    @abstractmethod
    def set_params(self, dict_of_params: dict):
        pass

    @abstractmethod
    def get_params(self):
        pass


class PyanNetModel(Model):
    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike],
        model_params: Optional[dict] = {
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        },
        finetune_params: Optional[dict] = None,
    ):
        super().__init__(checkpoint_path, model_params, finetune_params)
        self.model = md.from_pretrained(self.checkpoint_path)
        self.model_summary = self.model.summarize()

    def finetune(self):
        pass

    def predict(self, inference_audio_paths: list, output_path: str):
        pipeline = VoiceActivityDetection(segmentation=self.model)
        pipeline.instantiate(self.model_params)

        dict_of_prediction_status = {}

        for inference_audio_path in inference_audio_paths:
            list_of_start_times = []
            list_of_duration = []
            filename = Path(inference_audio_path).stem

            if not Path(inference_audio_path).exists():
                logger.error(f"File not found: {filename}")
                dict_of_prediction_status[inference_audio_path] = {
                    "status": "ERROR: File not found",
                    "rttm_path": None,
                }
                continue

            try:
                output = pipeline(inference_audio_path)
                hypothesis = output.get_timeline().support()

            except Exception:
                logger.error(f"Error inferring {filename}")
                dict_of_prediction_status[inference_audio_path] = {
                    "status": "ERROR: Error during model inference",
                    "rttm_path": None,
                }

            else:
                for line in hypothesis:
                    list_of_start_times.append(line.start)
                    list_of_duration.append(line.duration)

                status, output_rttm_path = generate_inference_rttm.generate_rttm(
                    list_of_start_times, list_of_duration, filename, output_path
                )

                if status != "SUCCESS":
                    logger.error(f"{filename} unable to write to rttm")
                    dict_of_prediction_status[inference_audio_path] = {
                        "status": status,
                        "rttm_path": None,
                    }

                else:
                    dict_of_prediction_status[inference_audio_path] = {
                        "status": status,
                        "rttm_path": output_rttm_path,
                    }

        return dict_of_prediction_status

    def set_params(self, dict_of_params: dict):
        self.model_params = dict_of_params

    def get_params(self):
        return self.model_params


#####
# For FastAPI testing
# test_model = PyanNetModel("/polyaxon-v1-data/workspaces/data_pyannote_600mins/lightning_logs/version_2/checkpoints/epoch=7.ckpt")

# list_of_audio = ["/polyaxon-v1-data/workspaces/data_pyannote_600mins/inference/inference_data/ES2015a.Array1-04_600_900.wav", "/polyaxon-v1-data/workspaces/data_pyannote_600mins/inference/inference_data/ES2015a.Array1-05_600_900.wav"]

# print(test_model.model_summary)
# dict_of_status = test_model.predict(list_of_audio, output_path= "/polyaxon-v1-data/workspaces/data_pyannote_600mins/output_test")
# print(dict_of_status)
