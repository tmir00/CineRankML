"""Tests for MLflow model registry helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from common.mlflow.registry import (
    assign_main_or_candidate_alias,
    promote_candidate_to_main,
    resolve_model_version_by_alias,
)


@patch("common.mlflow.registry._get_client")
def test_resolve_model_version_by_alias_reads_tag(mock_get_client: MagicMock) -> None:
    client = MagicMock()
    mock_get_client.return_value = client
    client.get_model_version_by_alias.return_value = MagicMock(
        tags={"model_version": "hybrid-v1-test"},
        run_id=None,
    )

    version = resolve_model_version_by_alias(
        tracking_uri="http://localhost:5000",
        registered_model_name="hybrid_ranker",
        alias="main",
    )
    assert version == "hybrid-v1-test"


@patch("common.mlflow.registry.set_model_alias")
@patch("common.mlflow.registry.register_hybrid_model_version")
@patch("common.mlflow.registry.resolve_model_version_by_alias")
def test_assign_main_when_no_main_alias(
    mock_resolve: MagicMock,
    mock_register: MagicMock,
    mock_set_alias: MagicMock,
) -> None:
    mock_resolve.return_value = None
    mock_register.return_value = 3

    alias = assign_main_or_candidate_alias(
        tracking_uri="http://localhost:5000",
        registered_model_name="hybrid_ranker",
        mlflow_run_id="run-123",
        model_version="hybrid-v1-test",
    )

    assert alias == "main"
    mock_set_alias.assert_called_once()


@patch("common.mlflow.registry._get_client")
@patch("common.mlflow.registry.set_model_alias")
@patch("common.mlflow.registry.resolve_model_version_by_alias")
def test_promote_candidate_to_main(
    mock_resolve: MagicMock,
    mock_set_alias: MagicMock,
    mock_get_client: MagicMock,
) -> None:
    mock_resolve.return_value = "hybrid-v2-test"
    client = MagicMock()
    mock_get_client.return_value = client
    client.get_model_version_by_alias.return_value = MagicMock(version="2")

    promoted = promote_candidate_to_main(
        tracking_uri="http://localhost:5000",
        registered_model_name="hybrid_ranker",
    )

    assert promoted == "hybrid-v2-test"
    client.delete_registered_model_alias.assert_called_once_with("hybrid_ranker", "candidate")
