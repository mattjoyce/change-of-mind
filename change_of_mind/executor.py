"""Mock executor for Phase 1 testing."""

from typing import Dict, Any
from .models import Model, ExecutionResult, RankingResult
import random


class MockExecutor:
    """
    Mock executor for Phase 1 testing.
    Simulates model responses without actual API calls.
    """

    # Template responses for different model types
    RESPONSE_TEMPLATES = {
        "coding": "Here's a {model_class} implementation:\n\n```python\n# Mock code response\n# (Phase 1 stub - no actual generation)\n```",
        "reasoning": "Based on {model_class} analysis:\n\n1. First consideration...\n2. Second point...\n\n(Phase 1 stub response)",
        "creative": "Once upon a time... (creative response from {model_class})\n\n(Phase 1 stub)",
        "default": "Response from {model_name} ({model_class}):\n\n(Phase 1 mock response - no actual API call)",
    }

    def execute(self, model: Model, message: str, context: Dict[str, Any] = None) -> ExecutionResult:
        """
        Simulate executing a query with the selected model.

        Args:
            model: Selected model
            message: User's message
            context: Optional context (unused in Phase 1)

        Returns:
            ExecutionResult with mock response
        """
        # Determine response type based on model tags
        response_type = "default"
        for tag in model.tags_list:
            if tag in self.RESPONSE_TEMPLATES:
                response_type = tag
                break

        # Generate mock response
        template = self.RESPONSE_TEMPLATES[response_type]
        response_text = template.format(model_name=model.name, model_class=model.class_tags)

        # Simulate confidence (higher quality models = higher confidence)
        confidence = 0.7 + (model.quality_rank * 0.25)
        confidence = min(1.0, confidence + random.uniform(-0.1, 0.1))

        return ExecutionResult(
            model_name=model.name,
            response_text=response_text,
            confidence=confidence,
            metadata={"mock": True, "message_length": len(message), "model_quality_rank": model.quality_rank},
            success=True,
        )

    def execute_with_routing(self, ranking_result: RankingResult, context: Dict[str, Any] = None) -> ExecutionResult:
        """
        Execute with the primary model from ranking result.

        Args:
            ranking_result: RankingResult from Router.route()
            context: Optional execution context

        Returns:
            ExecutionResult
        """
        return self.execute(ranking_result.primary.model, ranking_result.message, context)
