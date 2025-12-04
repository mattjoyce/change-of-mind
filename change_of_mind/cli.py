#!/usr/bin/env python3
"""
Command-line interface for LLM Model Router Phase 1 MVP.
"""
import argparse
import sys
from pathlib import Path
from .config import Config
from .router import Router
from .scorer import Scorer
from .matcher import KeywordMatcher
from .executor import MockExecutor


def main():
    parser = argparse.ArgumentParser(
        description="LLM Model Router - Route queries to optimal models based on persona preferences"
    )

    parser.add_argument("message", type=str, nargs="?", help="The message/query to route")

    parser.add_argument("-p", "--persona", type=str, help="Persona name (e.g., 'SysAdmin', 'Researcher', 'Creative')")

    parser.add_argument(
        "-c",
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing models.yaml and personas.yaml (default: config/)",
    )

    parser.add_argument("-n", "--top-n", type=int, default=3, help="Number of top models to display (default: 3)")

    parser.add_argument(
        "-e", "--execute", action="store_true", help="Execute the query with selected model (mock execution in Phase 1)"
    )

    parser.add_argument(
        "--explain", action="store_true", help="Show detailed scoring breakdown for all models (the working out)"
    )

    parser.add_argument("--list-personas", action="store_true", help="List available personas and exit")

    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    # Load configuration
    config = Config.from_directory(args.config_dir)

    # List personas
    if args.list_personas:
        print("Available Personas:")
        for name, persona in config.personas.items():
            print(f"\n  {name}:")
            print(f"    Quality: {persona.preference_quality:.2f}")
            print(f"    Cost: {persona.preference_cost:.2f}")
            print(f"    Privacy: {persona.preference_privacy:.2f}")
            print(f"    Speed: {persona.preference_speed:.2f}")
            print(f"    Stance: {persona.stance.value}")
        return 0

    # List models
    if args.list_models:
        print("Available Models:")
        for model in config.models:
            print(f"\n  {model.name}:")
            print(f"    Quality: {model.quality_rank:.2f}")
            print(f"    Cost: {model.cost_rank:.2f}")
            print(f"    Privacy: {model.privacy_rank:.2f}")
            print(f"    Speed: {model.speed_rank:.2f}")
            print(f"    Tags: {model.class_tags}")
        return 0

    # Validate required arguments for routing
    if not args.message or not args.persona:
        parser.error("message and --persona are required for routing")

    # Get persona
    persona = config.get_persona(args.persona)

    # Initialize router
    matcher = KeywordMatcher()
    scorer = Scorer(matcher)
    router = Router(config.models, scorer)

    # Route the message
    ranking = router.route(args.message, persona)

    # Display results
    if args.explain:
        print(router.explain_detailed(ranking))
    else:
        print(router.explain_decision(ranking, args.top_n))

    # Execute if requested
    if args.execute:
        print("\n" + "=" * 80)
        print("MOCK EXECUTION (Phase 1 - No actual API call)")
        print("=" * 80 + "\n")

        executor = MockExecutor()
        result = executor.execute_with_routing(ranking)

        print(f"Model: {result.model_name}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\nResponse:\n{result.response_text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
