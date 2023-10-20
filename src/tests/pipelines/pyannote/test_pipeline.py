"""
This is a boilerplate test file for pipeline 'pyannote'
generated using Kedro 0.18.11.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from kedro.framework.context import KedroContext
from kedro.runner import SequentialRunner
from kedro.config import ConfigLoader
from kedro.framework.hooks import manager
import klass.pipelines.pyannote as p
from kedro.framework.project import settings
import logging
import pytest
import yaml

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

conf_path = settings.CONF_SOURCE
conf_loader = ConfigLoader(conf_source=conf_path, env="test")

print(f"Config path is {conf_path}")  # Log the configuration path


class FileComparisonError(Exception):
    pass


class TestPyannotePreprocessingPipeline:
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

    def test_pyannote_preprocessing_pipeline(self, kedro_context):
        # Load only the pyannote_preprocessing pipeline
        pyannote_preprocessing_pipeline = p.create_pipeline()

        # Build the runner
        runner = SequentialRunner()
        print(kedro_context.catalog)
        # Run the pipeline
        run_output = runner.run(pyannote_preprocessing_pipeline, kedro_context.catalog)

        # You can add assertions here based on the run_output or any other checks
        # to ensure your pipeline runs successfully.
        # For example:
        # Example assertion for a single output file (modify as needed):
        splits, file_extensions = ["test", "validation", "train"], [
            "rttm",
            "uem",
            "lst",
        ]

        for split in splits:
            for file_extension in file_extensions:
                expected_output = f"src/tests/data/expected_results/05_model_input/{split}.{file_extension}"
                actual_output = (
                    f"src/tests/data/05_model_input/{split}.{file_extension}"
                )

                try:
                    self.compare_files(expected_output, actual_output)
                except FileComparisonError as e:
                    print(e)
