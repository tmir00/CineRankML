"""Tests for CF dataset prep settings and S3 client wiring."""

from __future__ import annotations

import pytest

from common.config.settings import CfDatasetSettings
from common.storage.s3 import create_s3_client


def test_create_s3_client_accepts_cf_dataset_settings() -> None:
    """CfDatasetSettings must work with the shared boto3 S3 client helper."""
    settings = CfDatasetSettings(
        s3_endpoint_url="http://localhost:9000",
        s3_access_key="minioadmin",
        s3_secret_key="minioadmin",
    )

    client = create_s3_client(settings)

    assert client.meta.service_model.service_name == "s3"


def test_cf_dataset_settings_reads_documented_env_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented .env.example names should populate CfDatasetSettings."""
    monkeypatch.setenv("CF_TRAIN_FRACTION", "0.75")
    monkeypatch.setenv("S3_SECRET_KEY", "secret-from-env")
    monkeypatch.setenv("CF_DATASET_METRICS_JOB_NAME", "prepare_cf_dataset_test")

    settings = CfDatasetSettings()

    assert settings.train_fraction == 0.75
    assert settings.s3_secret_key == "secret-from-env"
    assert settings.metrics_job_name == "prepare_cf_dataset_test"
