"""Scoring engine for computing model routing scores."""

from typing import List, Tuple
from .models import Model, Persona, ScoredModel
from .matcher import KeywordMatcher


class Scorer:
    """Computes routing scores for models based on persona preferences and heuristics."""

    def __init__(self, keyword_matcher: KeywordMatcher = None):
        self.matcher = keyword_matcher or KeywordMatcher()

    def compute_base_score(self, model: Model, persona: Persona) -> float:
        """
        Calculate weighted preference score.

        Formula:
            base_score = (
                persona.preference_quality * model.quality_rank +
                persona.preference_cost * (1 - model.cost_rank) +  # Inverted!
                persona.preference_privacy * model.privacy_rank +
                persona.preference_speed * model.speed_rank
            )

        Note: Cost is inverted - high cost preference means prefer LOW cost_rank models
        """
        score = (
            persona.preference_quality * model.quality_rank
            + persona.preference_cost * (1.0 - model.cost_rank)  # Cost inversion
            + persona.preference_privacy * model.privacy_rank
            + persona.preference_speed * model.speed_rank
        )

        # Normalize by sum of preferences to keep score in 0-1 range
        total_preference = (
            persona.preference_quality + persona.preference_cost + persona.preference_privacy + persona.preference_speed
        )

        if total_preference > 0:
            score = score / total_preference

        return score

    def compute_heuristic_score(
        self, message: str, model: Model, persona: Persona
    ) -> Tuple[float, List[str], List[str]]:
        """
        Apply task matching, keyword detection, and boost modifiers.

        Returns:
            Tuple of (heuristic_score, matched_keywords, applied_boosts)
        """
        heuristic_score = 0.0
        matched_keywords = []
        applied_boosts = []

        # 1. Task Type Matching
        matched_tasks = self.matcher.match_task_types(message, persona.preferred_task_types)

        if matched_tasks:
            # Check if model's tags align with matched task types
            for task in matched_tasks:
                normalized = self.matcher._normalize_task_type(task)
                if model.has_tag(normalized):
                    heuristic_score += 0.10  # Fixed boost for task alignment
                    matched_keywords.append(task)

        # 2. Direct Keyword Boost
        detected = self.matcher.detect_keywords(message)
        for task_type, keywords in detected.items():
            if model.has_tag(task_type):
                # Boost proportional to number of matching keywords (capped)
                boost = min(0.15, len(keywords) * 0.03)
                heuristic_score += boost
                matched_keywords.extend(keywords[:3])  # Limit for display

        # 3. Persona-Specific Boosts
        for boost_tag, boost_value in persona.boosts.items():
            if model.has_tag(boost_tag):
                heuristic_score += boost_value
                applied_boosts.append(f"{boost_tag}:{boost_value:+.2f}")

        # 4. Special Heuristics
        # Check for "local" preference in high-privacy personas
        if persona.preference_privacy > 0.8 and model.has_tag("local"):
            heuristic_score += 0.10
            applied_boosts.append("high_privacy_local:+0.10")

        return heuristic_score, matched_keywords, applied_boosts

    def score_all_models(self, message: str, persona: Persona, models: List[Model]) -> List[ScoredModel]:
        """
        Score all models and return sorted list.

        Args:
            message: User's input
            persona: Active persona
            models: Available models

        Returns:
            List of ScoredModel objects sorted by total_score (descending)
        """
        scored = []

        for model in models:
            base = self.compute_base_score(model, persona)
            heuristic, keywords, boosts = self.compute_heuristic_score(message, model, persona)
            total = base + heuristic

            scored.append(
                ScoredModel(
                    model=model,
                    base_score=base,
                    heuristic_score=heuristic,
                    total_score=total,
                    matched_keywords=keywords,
                    applied_boosts=boosts,
                )
            )

        # Sort by total score (descending)
        scored.sort(key=lambda x: x.total_score, reverse=True)
        return scored
