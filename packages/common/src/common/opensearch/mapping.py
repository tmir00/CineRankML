"""OpenSearch index mapping and alias management."""

from __future__ import annotations

import json

from pathlib import Path
from opensearchpy import OpenSearch
from common.config.settings import OpenSearchSettings

# This is the path to the file that contains the schema for the movies index.
_MAPPING_PATH = Path(__file__).resolve().parent / "movies_mapping.json"


def load_movies_mapping() -> dict:
    """
    Load the movie index mapping JSON bundled with this package.

    The canonical file lives next to mapping.py at
    packages/common/src/common/opensearch/movies_mapping.json.

    ============================ Returns ============================
    The parsed mapping body for index creation.
    """
    with _MAPPING_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def physical_index_name(alias: str, version: int) -> str:
    """ Return a versioned physical index name such as movies_v1. """
    return f"{alias}_v{version}"


def index_exists(client: OpenSearch, index_name: str) -> bool:
    """ Return True when the physical index already exists. """
    return bool(client.indices.exists(index=index_name))


def alias_exists(client: OpenSearch, alias: str) -> bool:
    """ Return True when the read alias is already defined. """
    return bool(client.indices.exists_alias(name=alias))


def get_alias_target_index(client: OpenSearch, alias: str) -> str | None:
    """
    Resolve which real index is currently pointed to by the given alias.

    ============================ Arguments ============================
    client: The OpenSearch client.
    alias: Stable alias name such as movies.

    ============================ Returns ============================
    The physical index name, or None when the alias does not exist.
    """
    # If the alias does not exist, return None.
    if not alias_exists(client, alias):
        return None
    
    # Get the alias target index.
    response = client.indices.get_alias(name=alias)
    
    # Get the list of indices that the alias is pointing to.
    indices = list(response.keys())
    return indices[0] if indices else None


def next_physical_index_version(client: OpenSearch, alias: str) -> int:
    """
    Pick the next physical index version for a rebuild.

    Do this by:
    1. Reading the current alias target when it exists.
    2. Incrementing the trailing version suffix.
    3. Defaulting to version 1 when no alias exists yet.

    ============================ Arguments ============================
    client: The OpenSearch client.
    alias: Stable alias name such as movies.

    ============================ Returns ============================
    The next version number to use for a physical index.
    """
    # Get the alias target index.
    target = get_alias_target_index(client, alias)
    
    # If the alias does not exist, return version 1.
    if target is None:
        return 1
    
    # Get the version suffix from the index name.
    suffix = target.rsplit("_v", maxsplit=1)[-1]
    
    # If the suffix is a number, return the next version number.
    if suffix.isdigit():
        return int(suffix) + 1
    
    # If the suffix is not a number, return version 1.
    return 1


def create_physical_index(client: OpenSearch, index_name: str) -> None:
    """
    Create one physical empty movie index when it does not already exist.

    ============================ Arguments ============================
    client: The OpenSearch client.
    index_name: Versioned physical index name such as movies_v1.
    """
    # If the index already exists, return.
    if index_exists(client, index_name):
        return
    
    # If the index does not exist, create it with the movies mapping.
    body = load_movies_mapping()
    client.indices.create(index=index_name, body=body)


def ensure_alias_points_to_index(client: OpenSearch, alias: str, index_name: str) -> None:
    """
    Point a read alias at one physical index, removing prior alias bindings.

    Do this by:
    1. Removing the prior alias binding when the target index is different.
    2. Adding the new alias binding to the index.

    ============================ Arguments ============================
    client: The OpenSearch client.
    alias: Stable alias name such as movies.
    index_name: Physical index that should receive reads.
    """
    # Build the list of actions to perform.
    actions: list[dict] = []

    # If the alias exists, get the current target index.
    if alias_exists(client, alias):
        current = get_alias_target_index(client, alias)

        # If the current target index is different from the new index, remove the prior alias binding.
        if current and current != index_name:
            actions.append({"remove": {"index": current, "alias": alias}})
    
    # Add the new alias binding to the index.
    actions.append({"add": {"index": index_name, "alias": alias}})

    # Perform the actions.
    client.indices.update_aliases(body={"actions": actions})


def ensure_movies_index(client: OpenSearch, settings: OpenSearchSettings) -> str:
    """
    Ensure the movies read alias exists and points at a physical index.

    Do this by:
    1. Creating movies_v1 when no alias exists yet.
    2. Returning the physical index name backing the alias.

    ============================ Arguments ============================
    client: The OpenSearch client.
    settings: OpenSearch index alias configuration.

    ============================ Returns ============================
    The physical index name used for writes during this run.
    """
    # Get the alias target index.
    alias = settings.opensearch_index_alias
    # If the alias's target index exists, return the target index.
    target = get_alias_target_index(client, alias)

    # If the alias's target index exists, return the target index.
    if target is not None:
        return target

    # If the alias's target index does not exist, create a new physical index.
    index_name = physical_index_name(alias, 1)
    create_physical_index(client, index_name)
    ensure_alias_points_to_index(client, alias, index_name)
    return index_name


def prepare_rebuild_index(client: OpenSearch, settings: OpenSearchSettings) -> str:
    """
    Create a new physical index for a full rebuild without swapping the alias yet.

    Do this by:
    1. Getting the next version number for the index.
    2. Creating a new physical index with the next version number.
    3. Returning the new index name.

    ============================ Arguments ============================
    client: The OpenSearch client.
    settings: OpenSearch index alias configuration.

    ============================ Returns ============================
    The new physical index name to bulk-index into.
    """
    # Get the alias and the next version number.
    alias = settings.opensearch_index_alias
    version = next_physical_index_version(client, alias)

    # Create a new physical index with the next version number.
    index_name = physical_index_name(alias, version)
    create_physical_index(client, index_name)
    return index_name


def finalize_rebuild_index(client: OpenSearch, settings: OpenSearchSettings, new_index_name: str) -> None:
    """
    Swap the movies alias to a rebuilt physical index and delete the old one.

    Do this by:
    1. Getting the alias and the old index.
    2. Pointing the alias to the new index.
    3. Deleting the old index when it is different from the new index and exists.

    ============================ Arguments ============================
    client: The OpenSearch client.
    settings: OpenSearch index alias configuration.
    new_index_name: Physical index that finished rebuilding.
    """
    # Get the alias and the old index.
    alias = settings.opensearch_index_alias
    # Get the old index.
    old_index = get_alias_target_index(client, alias)
    
    # Point the alias to the new index.
    ensure_alias_points_to_index(client, alias, new_index_name)
    # If the old index exists and is different from the new index, delete the old index.
    if old_index and old_index != new_index_name and index_exists(client, old_index):
        client.indices.delete(index=old_index)
