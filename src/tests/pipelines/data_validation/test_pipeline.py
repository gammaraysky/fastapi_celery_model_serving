"""
This is a boilerplate test file for pipeline 'data_validation'
generated using Kedro 0.18.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from kedro.framework.context import KedroContext
from kedro.runner import SequentialRunner
from kedro.config import ConfigLoader
from kedro.framework.hooks import manager
import klass.pipelines.data_validation as dv
from kedro.framework.project import settings
import json
import logging
import pytest
import yaml
import os

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

conf_path = settings.CONF_SOURCE
conf_loader = ConfigLoader(conf_source=conf_path, env="test")


class FileComparisonError(Exception):
    pass


class TestDataValidationPipeline:
    @pytest.fixture
    def kedro_context(self):
        context = KedroContext(
            package_name="klass",
            project_path="",
            config_loader=conf_loader,
            hook_manager=manager._create_hook_manager(),
        )
        return context

    @staticmethod
    def compare_files(expected_file, actual_file):
        with open(expected_file, "r") as expected_file, open(
            actual_file, "r"
        ) as actual_file:
            expected_content = expected_file.read()
            actual_content = actual_file.read()

        if expected_content != actual_content:
            raise FileComparisonError(
                f"Files {expected_file} and {actual_file} do not match."
            )

    @staticmethod
    def compare_json_files(expected_file, actual_file):
        with open(expected_file, "r") as expected_file, open(
            actual_file, "r"
        ) as actual_file:
            expected_content = json.load(expected_file)
            actual_content = json.load(actual_file)

        if expected_content != actual_content:
            raise FileComparisonError(
                f"Files {expected_file} and {actual_file} do not match."
            )

    def test_data_validation_pipeline(self, kedro_context):
        # Load only the pyannote_preprocessing pipeline
        data_validation_pipeline = dv.create_pipeline()

        # Build the runner
        runner = SequentialRunner()
        # Run the pipeline
        run_output = runner.run(data_validation_pipeline, kedro_context.catalog)

        for dataset in ["ami_far", "ali_far"]:
            for split in ["test", "train", "val"]:
                expected_folder_path = (
                    f"src/tests/data/expected_results/03_primary/{dataset}/{split}"
                )
                actual_folder_path = f"src/tests/data/03_primary/{dataset}/{split}"
                rttm_files = [
                    file
                    for file in os.listdir(expected_folder_path + "/rttm/")
                    if file.endswith(".rttm")
                ]
                for rttm_file in rttm_files:
                    expected_rttm = expected_folder_path + f"/rttm/{rttm_file}"
                    actual_rttm = actual_folder_path + f"/rttm/{rttm_file}"
                    try:
                        self.compare_files(expected_rttm, actual_rttm)
                    except FileComparisonError as e:
                        print(e)

                expected_json = expected_folder_path + "/format_check_report.json"
                actual_json = actual_folder_path + "/format_check_report.json"

                try:
                    self.compare_json_files(expected_json, actual_json)
                except FileComparisonError as e:
                    print(e)
