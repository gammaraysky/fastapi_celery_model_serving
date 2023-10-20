"""
This is a boilerplate test file for pipeline 'train_data_sampling'
generated using Kedro 0.18.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import sys
import os

sys.path.append("/polyaxon-v1-data/workspaces/hanafi/154/vad/klass/src/")
# print(os.getcwd())

from kedro.framework.context import KedroContext
from kedro.runner import SequentialRunner
from kedro.config import ConfigLoader
from kedro.framework.hooks import manager
import klass.pipelines.train_data_sampling as t_samp
import klass.pipelines.train_data_sampling.pipeline as pipeline

from kedro.framework.project import settings
import logging
import pytest
import yaml


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

conf_path = settings.CONF_SOURCE
print(conf_path)
conf_loader = ConfigLoader(conf_source=conf_path, env="test")

print(f"Config path is {conf_path}")  # Log the configuration path
# print(conf_loader)
# print(conf_loader["catalog"])
# print(conf_loader["parameters"])


class TestTrainDataSampling:
    @pytest.fixture
    def kedro_context(self):
        context = KedroContext(
            package_name="klass",
            project_path="",
            config_loader=conf_loader,
            hook_manager=manager._create_hook_manager(),
        )
        return context

    def test_train_data_sampling(self, kedro_context):
        # Load only the train_data_sampling pipeline
        train_data_sampling_pipeline = t_samp.create_pipeline()
