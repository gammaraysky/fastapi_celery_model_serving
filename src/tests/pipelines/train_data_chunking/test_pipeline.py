"""
This is a boilerplate test file for pipeline 'train_data_chunking'
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
import klass.pipelines.train_data_chunking as t_chunk
import klass.pipelines.train_data_chunking.pipeline as pipeline

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


class TestTrainDataChunking:
    # ali_ami_train_data_chunking_pipeline = pipeline.create_pipeline()

    @pytest.fixture
    def kedro_context(self):
        context = KedroContext(
            package_name="klass",
            project_path="",
            config_loader=conf_loader,
            hook_manager=manager._create_hook_manager(),
        )
        return context

    def test_train_data_chunking(self, kedro_context):
        # Load only the train_data_chunking pipeline
        train_data_chunking_pipeline = t_chunk.create_pipeline()

        # Build the runner
        runner = SequentialRunner()
        print(kedro_context.catalog)
        # Run the pipeline
        run_output = runner.run(train_data_chunking_pipeline, kedro_context.catalog)

        print(run_output)
        # You can add assertions here based on the run_output or any other checks
        # to ensure your pipeline runs successfully.
        # For example:
        # Example assertion for a single output file (modify as needed):
        namespaces = ["ali_far_train", "ami_far_train"]
        args = ["primary_wav_files", "primary_rttm_files", "train_chunking_config"]

        # splits, file_extensions = ["test", "validation", "train"], [
        #     "rttm",
        #     "uem",
        #     "lst",
        # ]

        # for split in splits:
        #     for file_extension in file_extensions:
        #         expected_output = f"src/tests/data/expected_results/05_model_input/{split}.{file_extension}"
        #         actual_output = (
        #             f"src/tests/data/05_model_input/{split}.{file_extension}"
        #         )
        #         with open(expected_output, "r") as expected_file, open(
        #             actual_output, "r"
        #         ) as actual_file:
        #             expected_content = expected_file.read()
        #             actual_content = actual_file.read()
        #         assert expected_content == actual_content
