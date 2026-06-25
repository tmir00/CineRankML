"""Bulk indexing helpers for OpenSearch."""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk


def bulk_index_movies(client: OpenSearch, index_name: str, 
                        documents: list[tuple[int, dict[str, Any]]]) -> tuple[int, list[dict[str, Any]]]:
    """
    Bulk upsert movie documents into one physical OpenSearch index.

    Do this by:
    1. Building bulk actions with movie_id as the document id.
    2. Sending the batch through the OpenSearch bulk helper.
    3. Returning how many documents succeeded and any per-item errors.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_name: Physical index to write into.
    documents: Pairs of (movie_id, document body).

    ============================ Returns ============================
    A tuple of (success_count, error_items).
    """
    # If there are no documents to index, return 0 successes and an empty list of errors.
    if not documents:
        return 0, []

    # Otherwise, build the list of bulk actions.
    actions = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": str(movie_id),
            "_source": body,
        }
        for movie_id, body in documents
    ]

    # Send the batch through the OpenSearch bulk helper.
    success_count, errors = bulk(client, actions, raise_on_error=False, raise_on_exception=False)
    
    # Return the number of successes and any errors.
    return int(success_count), list(errors)
