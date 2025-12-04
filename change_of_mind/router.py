"""Main routing engine for model selection."""

from typing import List
from .models import Model, Persona, RankingResult
from .scorer import Scorer
from .matcher import KeywordMatcher


class Router:
    """Main routing engine that selects optimal model for a given message and persona."""

    def __init__(self, models: List[Model], scorer: Scorer = None):
        self.models = models
        self.scorer = scorer or Scorer(KeywordMatcher())

    def route(self, message: str, persona: Persona) -> RankingResult:
        """
        Select the best model for the given message and persona.

        Args:
            message: User's input message
            persona: Active agent persona

        Returns:
            RankingResult with ranked models and scoring details
        """
        # Score all models
        scored_models = self.scorer.score_all_models(message, persona, self.models)

        return RankingResult(message=message, persona=persona, ranked_models=scored_models)

    def explain_decision(self, ranking: RankingResult, top_n: int = 3) -> str:
        """
        Generate human-readable explanation of routing decision.

        Args:
            ranking: RankingResult from route()
            top_n: Number of top models to include in explanation

        Returns:
            Formatted explanation string
        """
        lines = [
            f"Routing Decision for Persona: {ranking.persona.name}",
            f'Message: "{ranking.message[:80]}..."' if len(ranking.message) > 80 else f'Message: "{ranking.message}"',
            "",
            f"Top {top_n} Models:",
            "",
        ]

        for i, scored in enumerate(ranking.top_n(top_n), 1):
            lines.append(f"{i}. {scored.model.name}")
            lines.append(f"   Total Score: {scored.total_score:.3f}")
            lines.append(f"   - Base Score: {scored.base_score:.3f}")
            lines.append(f"   - Heuristic Score: {scored.heuristic_score:.3f}")

            if scored.matched_keywords:
                lines.append(f"   - Matched Keywords: {', '.join(scored.matched_keywords[:5])}")

            if scored.applied_boosts:
                lines.append(f"   - Applied Boosts: {', '.join(scored.applied_boosts)}")

            lines.append("")

        lines.append(f"SELECTED: {ranking.primary.model.name}")

        return "\n".join(lines)

    def explain_detailed(self, ranking: RankingResult) -> str:
        """
        Generate detailed explanation showing the working out for all models.

        Args:
            ranking: RankingResult from route()

        Returns:
            Detailed formatted explanation string with all scoring calculations
        """
        lines = [
            "=" * 80,
            "DETAILED ROUTING EXPLANATION",
            "=" * 80,
            "",
            f"Persona: {ranking.persona.name}",
            f'Message: "{ranking.message}"',
            "",
            "Persona Preferences:",
            f"  Quality:    {ranking.persona.preference_quality:.2f}",
            f"  Cost:       {ranking.persona.preference_cost:.2f} (high = prefer low cost)",
            f"  Privacy:    {ranking.persona.preference_privacy:.2f}",
            f"  Speed:      {ranking.persona.preference_speed:.2f}",
            f"  Creativity: {ranking.persona.preference_creativity:.2f}",
            "",
            "Persona Boosts:",
        ]

        if ranking.persona.boosts:
            for tag, boost in ranking.persona.boosts.items():
                lines.append(f"  {tag}: {boost:+.2f}")
        else:
            lines.append("  (none)")

        lines.extend(["", "=" * 80, "SCORING BREAKDOWN FOR ALL MODELS", "=" * 80, ""])

        for i, scored in enumerate(ranking.ranked_models, 1):
            model = scored.model
            lines.extend(
                [
                    f"\n{i}. {model.name}",
                    "-" * 80,
                    "",
                    "Model Characteristics:",
                    f"  Quality:  {model.quality_rank:.2f}",
                    f"  Cost:     {model.cost_rank:.2f}",
                    f"  Privacy:  {model.privacy_rank:.2f}",
                    f"  Speed:    {model.speed_rank:.2f}",
                    f"  Tags:     {model.class_tags}",
                    "",
                    "Base Score Calculation:",
                ]
            )

            # Show base score calculation
            persona = ranking.persona
            quality_component = persona.preference_quality * model.quality_rank
            cost_component = persona.preference_cost * (1.0 - model.cost_rank)
            privacy_component = persona.preference_privacy * model.privacy_rank
            speed_component = persona.preference_speed * model.speed_rank

            total_pref = (
                persona.preference_quality
                + persona.preference_cost
                + persona.preference_privacy
                + persona.preference_speed
            )

            lines.extend(
                [
                    f"  Quality:  {persona.preference_quality:.2f} × {model.quality_rank:.2f} = {quality_component:.3f}",
                    f"  Cost:     {persona.preference_cost:.2f} × (1 - {model.cost_rank:.2f}) = {cost_component:.3f}  [INVERTED]",
                    f"  Privacy:  {persona.preference_privacy:.2f} × {model.privacy_rank:.2f} = {privacy_component:.3f}",
                    f"  Speed:    {persona.preference_speed:.2f} × {model.speed_rank:.2f} = {speed_component:.3f}",
                    f"  ─────────────────────────────────────",
                    f"  Sum:      {quality_component + cost_component + privacy_component + speed_component:.3f}",
                    f"  Normalized by: {total_pref:.2f}",
                    f"  BASE SCORE: {scored.base_score:.3f}",
                    "",
                ]
            )

            # Show heuristic score breakdown
            lines.append("Heuristic Score Breakdown:")
            if scored.matched_keywords:
                lines.append(f"  Matched Keywords: {', '.join(scored.matched_keywords[:10])}")
            else:
                lines.append("  Matched Keywords: (none)")

            if scored.applied_boosts:
                lines.append("  Applied Boosts:")
                for boost in scored.applied_boosts:
                    lines.append(f"    • {boost}")
            else:
                lines.append("  Applied Boosts: (none)")

            lines.extend(
                [
                    f"  HEURISTIC SCORE: {scored.heuristic_score:.3f}",
                    "",
                    f"TOTAL SCORE: {scored.base_score:.3f} + {scored.heuristic_score:.3f} = {scored.total_score:.3f}",
                    "",
                ]
            )

        lines.extend(["=" * 80, f"FINAL SELECTION: {ranking.primary.model.name}", "=" * 80])

        return "\n".join(lines)
