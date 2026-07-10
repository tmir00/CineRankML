"""Tests for CF training settings and S3 client wiring."""

from __future__ import annotations

import pytest

from common.config.settings import CfTrainingSettings
from common.storage.s3 import create_s3_client


def test_create_s3_client_accepts_cf_training_settings() -> None:
    """CfTrainingSettings must work with the shared boto3 S3 client helper."""
    settings = CfTrainingSettings(
        s3_endpoint_url="http://localhost:9000",
        s3_access_key="minioadmin",
        s3_secret_key="minioadmin",
    )

    client = create_s3_client(settings)

    assert client.meta.service_model.service_name == "s3"


def test_cf_training_settings_reads_documented_env_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented .env.example names should populate CfTrainingSettings."""
    monkeypatch.setenv("CF_EPOCHS", "10")
    monkeypatch.setenv("CF_EARLY_STOPPING_PATIENCE", "0")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "cf_test")
    monkeypatch.setenv("CF_METRICS_JOB_NAME", "train_cf_test")

    settings = CfTrainingSettings()

    assert settings.num_epochs == 10
    assert settings.early_stopping_patience == 0
    assert settings.mlflow_experiment_name == "cf_test"
    assert settings.metrics_job_name == "train_cf_test"
