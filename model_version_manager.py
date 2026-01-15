"""
Model Version Manager for saving and loading versioned checkpoints.

Manages a pool of model versions for evaluation and training against past versions.
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch


@dataclass
class ModelVersion:
    """Represents a saved model version."""
    version_id: int
    path: str
    timesteps: int
    created_at: float  # Unix timestamp


class ModelVersionManager:
    """
    Manages versioned model checkpoints for evaluation and training.

    Features:
    - Saves model versions at configurable intervals
    - Maintains a pool of past versions for eval/training
    - Tracks metadata (timesteps, creation time) for each version
    """

    def __init__(
        self,
        versions_dir: str,
        max_versions: int = 20,
        save_every_ts: int = 5_000_000,
    ):
        """
        Args:
            versions_dir: Directory to store versioned checkpoints
            max_versions: Maximum number of versions to keep (oldest pruned)
            save_every_ts: Save a new version every N timesteps
        """
        self.versions_dir = versions_dir
        self.max_versions = max_versions
        self.save_every_ts = save_every_ts

        self.versions: Dict[int, ModelVersion] = {}
        self.next_version_id = 0
        self.last_save_ts = 0

        os.makedirs(versions_dir, exist_ok=True)
        self._load_existing_versions()

    def _load_existing_versions(self) -> None:
        """Load existing versions from disk."""
        metadata_path = os.path.join(self.versions_dir, "versions.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                data = json.load(f)
                self.next_version_id = data.get("next_version_id", 0)
                self.last_save_ts = data.get("last_save_ts", 0)
                for v in data.get("versions", []):
                    version = ModelVersion(
                        version_id=v["version_id"],
                        path=v["path"],
                        timesteps=v["timesteps"],
                        created_at=v["created_at"]
                    )
                    if os.path.exists(version.path):
                        self.versions[version.version_id] = version

    def _save_metadata(self) -> None:
        """Save versions metadata to disk."""
        metadata_path = os.path.join(self.versions_dir, "versions.json")
        data = {
            "next_version_id": self.next_version_id,
            "last_save_ts": self.last_save_ts,
            "versions": [
                {
                    "version_id": v.version_id,
                    "path": v.path,
                    "timesteps": v.timesteps,
                    "created_at": v.created_at
                }
                for v in self.versions.values()
            ]
        }
        with open(metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def should_save_version(self, cumulative_timesteps: int) -> bool:
        """Check if it's time to save a new version."""
        return cumulative_timesteps - self.last_save_ts >= self.save_every_ts

    def save_version(
        self,
        actor_state_dict: dict,
        cumulative_timesteps: int,
        critic_state_dict: Optional[dict] = None,
    ) -> ModelVersion:
        """
        Save a new model version.

        Args:
            actor_state_dict: Actor model state dict
            cumulative_timesteps: Current training timesteps
            critic_state_dict: Optional critic state dict

        Returns:
            The created ModelVersion
        """
        import time

        version_id = self.next_version_id
        version_path = os.path.join(self.versions_dir, f"v{version_id}")
        os.makedirs(version_path, exist_ok=True)

        # Save actor
        torch.save(actor_state_dict, os.path.join(version_path, "actor.pt"))

        # Optionally save critic
        if critic_state_dict is not None:
            torch.save(critic_state_dict, os.path.join(version_path, "critic.pt"))

        version = ModelVersion(
            version_id=version_id,
            path=version_path,
            timesteps=cumulative_timesteps,
            created_at=time.time()
        )

        self.versions[version_id] = version
        self.next_version_id += 1
        self.last_save_ts = cumulative_timesteps

        # Prune old versions if needed
        self._prune_old_versions()

        self._save_metadata()

        print(f"Saved model version v{version_id} at {cumulative_timesteps} timesteps")
        return version

    def _prune_old_versions(self) -> None:
        """Remove oldest versions if over max_versions limit."""
        if len(self.versions) > self.max_versions:
            # Sort by version_id (oldest first)
            sorted_versions = sorted(self.versions.values(), key=lambda v: v.version_id)
            versions_to_remove = sorted_versions[:-self.max_versions]

            for version in versions_to_remove:
                if os.path.exists(version.path):
                    shutil.rmtree(version.path)
                del self.versions[version.version_id]

    def get_version(self, version_id: int) -> Optional[ModelVersion]:
        """Get a specific version by ID."""
        return self.versions.get(version_id)

    def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the most recent version."""
        if not self.versions:
            return None
        return max(self.versions.values(), key=lambda v: v.version_id)

    def get_all_versions(self) -> List[ModelVersion]:
        """Get all available versions sorted by ID."""
        return sorted(self.versions.values(), key=lambda v: v.version_id)

    def get_random_version(self) -> Optional[ModelVersion]:
        """Get a random version from the pool."""
        import random
        if not self.versions:
            return None
        return random.choice(list(self.versions.values()))

    def load_actor(self, version_id: int, device: str = "cpu") -> Optional[dict]:
        """Load actor state dict for a version."""
        version = self.get_version(version_id)
        if version is None:
            return None
        actor_path = os.path.join(version.path, "actor.pt")
        if not os.path.exists(actor_path):
            return None
        return torch.load(actor_path, map_location=device, weights_only=True)

    def get_version_ids(self) -> List[int]:
        """Get list of all version IDs."""
        return sorted(self.versions.keys())
