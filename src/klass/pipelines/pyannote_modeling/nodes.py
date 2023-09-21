"""
This is a boilerplate pipeline 'pyannote_modeling'
generated using Kedro 0.18.11
"""
import logging

import mlflow
import pytorch_lightning as pl
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.tasks import VoiceActivityDetection

# for Linux need to "sudo apt-get install libsndfile1"
from pyannote.database import FileFinder, registry
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torchmetrics.classification import BinaryF1Score

def train_voice_activity_detection(
    seed: int,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    database_config: str,
    vad_config: dict,
    early_stopping_config: dict,
    trainer_config: dict,
    checkpoint_config: dict,
    MLFLOW: bool,
) -> None:
    """Train Voice Activity Detection Model

    Args:
        seed (int): Random seed for reproducibility.
        mlflow_tracking_uri (str): Cluster IP of the MLFlow service.
        mlflow_experiment_name (str): Name of the MLFlow experiment.
        database_config (str): Path to the database configuration file.
        vad_config (dict): Configuration for Voice Activity Detection (VAD) model.
        early_stopping_config (dict): Configuration for early stopping callback.
        checkpoint_config (dict): Configuration for saving model checkpoints.
        trainer_config (dict): Configuration for PyTorch Lightning Trainer.
        MLFLOW (bool): True for MLFlow logging.
    Returns:
        None
    """

    seed_everything(seed=seed, workers=True)
    if MLFLOW:
        mlflow.set_tracking_uri(
            mlflow_tracking_uri
        )  # This is the Cluster IP of the MLFlow service
        mlflow.set_experiment(mlflow_experiment_name)
        mlflow.start_run()

    registry.load_database(
        database_config, mode="LoadingMode=LoadingMode.OVERRIDE"
    )
    # pyannote.database protocol
    protocol = registry.get_protocol(
        vad_config["protocol"], 
        preprocessors={"audio": FileFinder()}
    )

    vad_task = VoiceActivityDetection(
        protocol, 
        duration=vad_config.get("duration", 2.0), 
        batch_size=vad_config.get("batch_size", 128), 
        metric=BinaryF1Score()
    )

    vad_model = PyanNet(
        task=vad_task, 
        sincnet={"stride": vad_config.get("stride", 10)}
    )

    monitor, direction = vad_task.val_monitor

    print(monitor, direction)

    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=checkpoint_config.get("save_top_k", -1),
        every_n_epochs=checkpoint_config.get("every_n_epochs", 1),
        save_last=checkpoint_config.get("save_last", False),
        save_weights_only=checkpoint_config.get("save_weights_only", False),
        filename="{epoch}",
        verbose=checkpoint_config.get("verbose", False),
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=direction,
        min_delta=early_stopping_config.get("min_delta", 0.0),
        patience=early_stopping_config.get("patience", 5),
        strict=True,
        verbose=early_stopping_config.get("verbose", False),
    )

    callbacks = [checkpoint, early_stopping]

    trainer = pl.Trainer(
        devices=trainer_config.get("devices", 1),
        accelerator=trainer_config.get("accelerator", "auto"),
        max_epochs=trainer_config.get("max_epochs", 100),
        callbacks=callbacks,
        deterministic=True,
    )
    if MLFLOW:
        mlflow.pytorch.autolog()

    trainer.fit(vad_model)

    if MLFLOW:
        mlflow.end_run()
