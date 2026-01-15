"""
Opponent Pool for training against past model versions.

Manages a pool of past model checkpoints that can be used as opponents
during training, enabling curriculum learning and more robust policies.
"""

import os
import random
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .model_version_manager import ModelVersionManager
from .trueskill_tracker import TrueSkillTracker, TRUESKILL_AVAILABLE

if TRUESKILL_AVAILABLE:
    from trueskill import Rating


@dataclass
class Opponent:
    """Represents a loaded opponent model."""
    version_id: int
    policy: Any  # The loaded policy object
    rating_mu: float = 25.0
    rating_sigma: float = 8.333


class OpponentPool:
    """
    Manages a pool of opponent models for training.

    Features:
    - Loads and caches past model versions
    - Selects opponents based on TrueSkill ratings or random sampling
    - Supports probability-weighted selection for curriculum learning
    """

    def __init__(
        self,
        version_manager: ModelVersionManager,
        policy_class: type,
        policy_kwargs: dict,
        device: str = "cpu",
        max_cached_opponents: int = 5,
        trueskill_tracker: Optional[TrueSkillTracker] = None,
    ):
        """
        Args:
            version_manager: Manager for saved model versions
            policy_class: Class to instantiate for opponent policies
            policy_kwargs: Kwargs for policy constructor
            device: Device to load models on
            max_cached_opponents: Max opponents to keep in memory
            trueskill_tracker: Optional tracker for rating-based selection
        """
        self.version_manager = version_manager
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.max_cached_opponents = max_cached_opponents
        self.trueskill_tracker = trueskill_tracker

        # Cache of loaded opponents: version_id -> Opponent
        self.cached_opponents: Dict[int, Opponent] = {}
        self.cache_order: List[int] = []  # LRU order

    def _load_opponent(self, version_id: int) -> Optional[Opponent]:
        """Load an opponent from disk."""
        state_dict = self.version_manager.load_actor(version_id, self.device)
        if state_dict is None:
            return None

        policy = self.policy_class(**self.policy_kwargs, device=self.device)
        policy.load_state_dict(state_dict)
        policy.to(self.device)
        policy.eval()

        # Get rating if tracker available
        mu, sigma = 25.0, 8.333
        if self.trueskill_tracker:
            version_key = f"v{version_id}"
            if version_key in self.trueskill_tracker.ratings:
                rating = self.trueskill_tracker.ratings[version_key]
                mu, sigma = rating.mu, rating.sigma

        return Opponent(
            version_id=version_id,
            policy=policy,
            rating_mu=mu,
            rating_sigma=sigma
        )

    def get_opponent(self, version_id: int) -> Optional[Opponent]:
        """Get an opponent, loading from cache or disk."""
        if version_id in self.cached_opponents:
            # Move to end of LRU order
            self.cache_order.remove(version_id)
            self.cache_order.append(version_id)
            return self.cached_opponents[version_id]

        # Load from disk
        opponent = self._load_opponent(version_id)
        if opponent is None:
            return None

        # Add to cache
        self.cached_opponents[version_id] = opponent
        self.cache_order.append(version_id)

        # Evict oldest if over limit
        while len(self.cache_order) > self.max_cached_opponents:
            oldest = self.cache_order.pop(0)
            del self.cached_opponents[oldest]

        return opponent

    def get_random_opponent(self) -> Optional[Opponent]:
        """Get a random opponent from available versions."""
        version_ids = self.version_manager.get_version_ids()
        if not version_ids:
            return None
        version_id = random.choice(version_ids)
        return self.get_opponent(version_id)

    def select_opponent_by_skill(
        self,
        current_mu: float = 25.0,
        current_sigma: float = 8.333,
        prefer_similar_skill: bool = True,
    ) -> Optional[Opponent]:
        """
        Select an opponent based on skill matching.

        Args:
            current_mu: Current model's TrueSkill mu
            current_sigma: Current model's TrueSkill sigma
            prefer_similar_skill: If True, prefer opponents with similar skill

        Returns:
            Selected opponent or None if no versions available
        """
        version_ids = self.version_manager.get_version_ids()
        if not version_ids:
            return None

        if not TRUESKILL_AVAILABLE or self.trueskill_tracker is None:
            # Fall back to random selection
            return self.get_random_opponent()

        # Calculate selection probabilities based on match quality
        probabilities = []
        for vid in version_ids:
            version_key = f"v{vid}"
            if version_key in self.trueskill_tracker.ratings:
                rating = self.trueskill_tracker.ratings[version_key]
                opp_mu, opp_sigma = rating.mu, rating.sigma
            else:
                opp_mu, opp_sigma = 25.0, 8.333

            if prefer_similar_skill:
                # Higher probability for opponents with similar skill
                # Use match quality formula from TrueSkill
                skill_diff = abs(current_mu - opp_mu)
                combined_sigma = (current_sigma**2 + opp_sigma**2) ** 0.5
                # Probability peaks when skill_diff is 0
                prob = max(0.01, 1.0 / (1.0 + skill_diff / combined_sigma))
            else:
                # Uniform probability but weighted by uncertainty
                # Prefer opponents with high sigma (need more games)
                prob = max(0.01, opp_sigma / 8.333)

            probabilities.append(prob)

        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Select based on probabilities
        selected_id = random.choices(version_ids, weights=probabilities, k=1)[0]
        return self.get_opponent(selected_id)

    def select_high_uncertainty_opponent(self, sigma_threshold: float = 4.0) -> Optional[Opponent]:
        """
        Select an opponent with high rating uncertainty.

        This is useful for evaluation - playing against uncertain opponents
        helps establish their true skill level.

        Args:
            sigma_threshold: Minimum sigma to consider "high uncertainty"

        Returns:
            Selected opponent or None
        """
        if not self.trueskill_tracker:
            return self.get_random_opponent()

        version_ids = self.version_manager.get_version_ids()
        high_uncertainty = []

        for vid in version_ids:
            version_key = f"v{vid}"
            if version_key in self.trueskill_tracker.ratings:
                rating = self.trueskill_tracker.ratings[version_key]
                if rating.sigma >= sigma_threshold:
                    high_uncertainty.append((vid, rating.sigma))
            else:
                # Unknown versions have default high uncertainty
                high_uncertainty.append((vid, 8.333))

        if not high_uncertainty:
            return self.get_random_opponent()

        # Weight by sigma (higher uncertainty = more likely to select)
        weights = [sigma for _, sigma in high_uncertainty]
        selected_id = random.choices(
            [vid for vid, _ in high_uncertainty],
            weights=weights,
            k=1
        )[0]

        return self.get_opponent(selected_id)

    def get_available_versions(self) -> List[int]:
        """Get list of available opponent version IDs."""
        return self.version_manager.get_version_ids()

    def clear_cache(self) -> None:
        """Clear the opponent cache."""
        self.cached_opponents.clear()
        self.cache_order.clear()


class MixedOpponentAgentController:
    """
    Still in development.
    Wrapper that enables training with mixed opponents.

    This wraps an existing agent controller to allow some agents
    to be controlled by past model versions instead of the current model.

    Usage:
        # In your training setup
        mixed_controller = MixedOpponentAgentController(
            main_controller=your_ppo_controller,
            opponent_pool=opponent_pool,
            opponent_probability=0.2,  # 20% chance to play vs past version
        )
    """

    def __init__(
        self,
        main_controller: Any,
        opponent_pool: OpponentPool,
        opponent_probability: float = 0.2,
        opponent_team: str = "orange",  # Which team uses opponents
    ):
        """
        Args:
            main_controller: The main PPO agent controller
            opponent_pool: Pool of past model versions
            opponent_probability: Probability of using past opponent (0-1)
            opponent_team: Which team to replace ("orange", "blue", or "random")
        """
        self.main_controller = main_controller
        self.opponent_pool = opponent_pool
        self.opponent_probability = opponent_probability
        self.opponent_team = opponent_team

        # Current opponent for this episode (if any)
        self.current_opponent: Optional[Opponent] = None
        self.using_opponent = False

    def on_episode_start(self) -> None:
        """Called at the start of each episode to decide opponent."""
        if random.random() < self.opponent_probability:
            self.current_opponent = self.opponent_pool.select_opponent_by_skill()
            self.using_opponent = self.current_opponent is not None
        else:
            self.current_opponent = None
            self.using_opponent = False

    def get_actions(self, agent_id_list: List, obs_list: List) -> Tuple[List, Any]:
        """
        Get actions, using opponent for designated team.

        Note: This is a simplified implementation. In practice, you'd need
        to integrate this more deeply with the agent controller's action loop.
        """
        if not self.using_opponent or self.current_opponent is None:
            return self.main_controller.get_actions(agent_id_list, obs_list)

        # Split agents by team and get appropriate actions
        # This is a simplified version - actual implementation depends
        # on how agent IDs encode team membership in your setup
        main_actions, main_log_probs = self.main_controller.get_actions(
            agent_id_list, obs_list
        )

        # In a full implementation, you would:
        # 1. Identify which agents belong to the opponent team
        # 2. Get actions for those agents from self.current_opponent.policy
        # 3. Merge the actions appropriately

        return main_actions, main_log_probs