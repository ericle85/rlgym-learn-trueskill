"""
TrueSkill rating tracker for model versions.

Tracks model performance over time using TrueSkill ratings from evaluation games.
Supports per-gamemode ratings (1v1, 2v2, 3v3).
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import os

try:
    from trueskill import Rating, rate, SIGMA as TRUESKILL_SIGMA
    TRUESKILL_AVAILABLE = True
except ImportError:
    TRUESKILL_AVAILABLE = False
    Rating = None
    TRUESKILL_SIGMA = 8.333


@dataclass
class GameOutcome:
    """Represents the outcome of a single game."""
    blue_goals: int
    orange_goals: int
    blue_agents: List[str]
    orange_agents: List[str]
    gamemode: str = "1v1"
    game_length_seconds: float = 0.0

    @property
    def result(self) -> int:
        """Returns: 1 for blue win, -1 for orange win, 0 for draw."""
        if self.blue_goals > self.orange_goals:
            return 1
        elif self.orange_goals > self.blue_goals:
            return -1
        return 0


@dataclass
class ModelRating:
    """TrueSkill rating for a model version."""
    mu: float = 25.0
    sigma: float = 8.333
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def conservative_rating(self) -> float:
        """Conservative skill estimate (mu - 3*sigma)."""
        return self.mu - 3 * self.sigma

    def to_dict(self) -> dict:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelRating':
        return cls(**d)


@dataclass
class VersionRatings:
    """Per-gamemode ratings for a model version."""
    ratings: Dict[str, ModelRating] = field(default_factory=dict)

    def get_rating(self, gamemode: str) -> ModelRating:
        """Get or create rating for gamemode."""
        if gamemode not in self.ratings:
            self.ratings[gamemode] = ModelRating()
        return self.ratings[gamemode]

    def to_dict(self) -> dict:
        return {gm: r.to_dict() for gm, r in self.ratings.items()}

    @classmethod
    def from_dict(cls, d: dict) -> 'VersionRatings':
        return cls(ratings={gm: ModelRating.from_dict(r) for gm, r in d.items()})


class TrueSkillTracker:
    """
    Tracks TrueSkill ratings for model versions per gamemode.

    Updates ratings based on evaluation game results between
    the current model and past checkpoints.

    Supports separate ratings for 1v1, 2v2, 3v3 gamemodes.
    Tracks deterministic and stochastic eval modes separately.
    """

    GAMEMODES = ["1v1", "2v2", "3v3"]
    EVAL_MODES = ["deterministic", "stochastic"]

    def __init__(self, min_sigma: float = 1.0):
        """
        Args:
            min_sigma: Minimum sigma to prevent overconfident ratings
        """
        self.min_sigma = min_sigma

        # version_id -> VersionRatings (contains per-gamemode ratings)
        self.version_ratings: Dict[str, VersionRatings] = {}

        # Initialize "current" model rating
        self.version_ratings["current"] = VersionRatings()

        # Total eval games played per gamemode
        self.total_eval_games: Dict[str, int] = {gm: 0 for gm in self.GAMEMODES}

        # History for plotting: (game_num, version, gamemode, mu, sigma)
        self.rating_history: List[Tuple[int, str, str, float, float]] = []

    @staticmethod
    def _make_rating_key(gamemode: str, deterministic: bool = True) -> str:
        """Create internal key combining gamemode and eval mode."""
        eval_mode = "deterministic" if deterministic else "stochastic"
        return f"{gamemode}_{eval_mode}"

    @staticmethod
    def _parse_rating_key(key: str) -> Tuple[str, str]:
        """Parse internal key into (gamemode, eval_mode)."""
        for gm in TrueSkillTracker.GAMEMODES:
            for em in TrueSkillTracker.EVAL_MODES:
                if key == f"{gm}_{em}":
                    return gm, em
        # Fallback for old-style keys (just gamemode)
        if key in TrueSkillTracker.GAMEMODES:
            return key, "deterministic"
        return key, "deterministic"

    def get_or_create_rating(self, version: str, gamemode: str) -> ModelRating:
        """Get rating for version and gamemode, creating if doesn't exist."""
        if version not in self.version_ratings:
            # New versions inherit from current model's rating with increased uncertainty
            current = self.version_ratings.get("current", VersionRatings())
            current_rating = current.get_rating(gamemode)
            new_ratings = VersionRatings()
            new_ratings.ratings[gamemode] = ModelRating(
                mu=current_rating.mu,
                sigma=max(current_rating.sigma, TRUESKILL_SIGMA / 2)
            )
            self.version_ratings[version] = new_ratings
        return self.version_ratings[version].get_rating(gamemode)

    def update_ratings(
        self,
        blue_version: str,
        orange_version: str,
        result: int,
        gamemode: str = "1v1",
        deterministic: bool = True
    ) -> Dict[str, ModelRating]:
        """
        Update TrueSkill ratings based on a match result.

        Args:
            blue_version: Version ID of blue team (e.g., "v0", "current")
            orange_version: Version ID of orange team
            result: 1 for blue win, -1 for orange win, 0 for draw
            gamemode: Game mode ("1v1", "2v2", "3v3")
            deterministic: Whether this was a deterministic or stochastic eval

        Returns:
            Dict of updated ratings
        """
        if not TRUESKILL_AVAILABLE:
            print("TrueSkill not available. Install with: pip install trueskill")
            return {}

        # Create rating key combining gamemode and eval mode
        rating_key = self._make_rating_key(gamemode, deterministic)

        # Get or create ratings for this key
        blue_rating = self.get_or_create_rating(blue_version, rating_key)
        orange_rating = self.get_or_create_rating(orange_version, rating_key)

        # Convert to trueskill Rating objects
        blue_ts = Rating(mu=blue_rating.mu, sigma=blue_rating.sigma)
        orange_ts = Rating(mu=orange_rating.mu, sigma=orange_rating.sigma)

        # Determine ranks (0 = winner, 1 = loser, same = draw)
        if result == 1:  # Blue wins
            ranks = [0, 1]
            blue_rating.wins += 1
            orange_rating.losses += 1
        elif result == -1:  # Orange wins
            ranks = [1, 0]
            blue_rating.losses += 1
            orange_rating.wins += 1
        else:  # Draw
            ranks = [0, 0]
            blue_rating.draws += 1
            orange_rating.draws += 1

        # Calculate new ratings
        (new_blue,), (new_orange,) = rate([(blue_ts,), (orange_ts,)], ranks=ranks)

        # Apply minimum sigma and update
        blue_rating.mu = new_blue.mu
        blue_rating.sigma = max(new_blue.sigma, self.min_sigma)
        blue_rating.games_played += 1

        orange_rating.mu = new_orange.mu
        orange_rating.sigma = max(new_orange.sigma, self.min_sigma)
        orange_rating.games_played += 1

        self.total_eval_games[gamemode] = self.total_eval_games.get(gamemode, 0) + 1
        total_games = sum(self.total_eval_games.values())

        # Record history (use rating_key which includes eval mode)
        self.rating_history.append((
            total_games, blue_version, rating_key, blue_rating.mu, blue_rating.sigma
        ))
        self.rating_history.append((
            total_games, orange_version, rating_key, orange_rating.mu, orange_rating.sigma
        ))

        return {blue_version: blue_rating, orange_version: orange_rating}

    def promote_current_to_version(self, version_id: int) -> None:
        """
        When saving a new checkpoint, create a version entry from current.

        The 'current' rating represents the live training model.
        When we save a checkpoint, we copy its rating to the version.
        """
        version_key = f"v{version_id}"
        current = self.version_ratings.get("current", VersionRatings())

        # Create version with current's ratings but increased sigma
        new_version = VersionRatings()
        for gamemode, rating in current.ratings.items():
            new_version.ratings[gamemode] = ModelRating(
                mu=rating.mu,
                sigma=max(rating.sigma, TRUESKILL_SIGMA / 3),
                games_played=0,
                wins=0,
                losses=0,
                draws=0
            )
        self.version_ratings[version_key] = new_version

    def get_metrics(self, gamemode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics dict for logging to wandb.

        Returns only Rating per gamemode for deterministic and stochastic eval modes.

        Args:
            gamemode: If specified, only return metrics for this gamemode.
                     If None, return metrics for all gamemodes.
        """
        metrics = {}

        gamemodes = [gamemode] if gamemode else self.GAMEMODES
        current = self.version_ratings.get("current")

        if not current:
            return metrics

        for gm in gamemodes:
            # Check both deterministic and stochastic ratings
            for deterministic in [True, False]:
                rating_key = self._make_rating_key(gm, deterministic)
                eval_name = "Deterministic" if deterministic else "Stochastic"

                if rating_key in current.ratings:
                    rating = current.ratings[rating_key]
                    # Format: "Rating {gamemode}/eval_mode" so wandb groups both modes together
                    metrics[f"Rating {gm}/{eval_name}"] = rating.mu

        return metrics

    def get_leaderboard(
        self, gamemode: str = "1v1", deterministic: bool = True
    ) -> List[Tuple[str, ModelRating]]:
        """Get versions sorted by conservative rating (best first) for a gamemode."""
        rating_key = self._make_rating_key(gamemode, deterministic)
        rated = []
        for k, v in self.version_ratings.items():
            # Check both new format (with eval mode) and old format (just gamemode)
            if rating_key in v.ratings and v.ratings[rating_key].games_played > 0:
                rated.append((k, v.ratings[rating_key]))
            elif gamemode in v.ratings and v.ratings[gamemode].games_played > 0:
                # Fallback for old format
                rated.append((k, v.ratings[gamemode]))
        return sorted(rated, key=lambda x: x[1].conservative_rating, reverse=True)

    def save(self, path: str) -> None:
        """Save tracker state to file."""
        state = {
            "min_sigma": self.min_sigma,
            "total_eval_games": self.total_eval_games,
            "version_ratings": {
                k: v.to_dict() for k, v in self.version_ratings.items()
            },
            "rating_history": self.rating_history,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load tracker state from file."""
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            state = json.load(f)

        self.min_sigma = state.get("min_sigma", 1.0)

        # Handle migration from old format (int) to new format (dict per gamemode)
        loaded_eval_games = state.get("total_eval_games", {gm: 0 for gm in self.GAMEMODES})
        if isinstance(loaded_eval_games, int):
            # Old format was a single int, migrate to dict (assume 1v1)
            self.total_eval_games = {gm: 0 for gm in self.GAMEMODES}
            self.total_eval_games["1v1"] = loaded_eval_games
        else:
            self.total_eval_games = loaded_eval_games

        # Handle both old format (flat ratings) and new format (per-gamemode)
        if "version_ratings" in state:
            self.version_ratings = {
                k: VersionRatings.from_dict(v)
                for k, v in state.get("version_ratings", {}).items()
            }
        elif "ratings" in state:
            # Migrate old format to new format (assume 1v1)
            self.version_ratings = {}
            for k, v in state.get("ratings", {}).items():
                vr = VersionRatings()
                vr.ratings["1v1"] = ModelRating.from_dict(v)
                self.version_ratings[k] = vr

        self.rating_history = [tuple(x) for x in state.get("rating_history", [])]

        # Ensure current exists
        if "current" not in self.version_ratings:
            self.version_ratings["current"] = VersionRatings()

    def print_leaderboard(
        self, gamemode: Optional[str] = None, deterministic: bool = True
    ) -> None:
        """Print a formatted leaderboard."""
        gamemodes = [gamemode] if gamemode else self.GAMEMODES
        eval_mode = "Deterministic" if deterministic else "Stochastic"

        for gm in gamemodes:
            leaderboard = self.get_leaderboard(gm, deterministic)
            if not leaderboard:
                continue

            print(f"\n=== TrueSkill Leaderboard ({gm} - {eval_mode}) ===")
            print(f"{'Rank':<6}{'Version':<12}{'Rating':<10}{'Mu':<10}{'Sigma':<10}{'W/L/D':<15}")
            print("-" * 63)

            for rank, (version, rating) in enumerate(leaderboard, 1):
                wld = f"{rating.wins}/{rating.losses}/{rating.draws}"
                print(
                    f"{rank:<6}{version:<12}{rating.conservative_rating:<10.2f}"
                    f"{rating.mu:<10.2f}{rating.sigma:<10.2f}{wld:<15}"
                )
        print()
