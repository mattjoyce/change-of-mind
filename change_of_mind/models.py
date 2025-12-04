"""Data models for the LLM Model Router."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from enum import Enum


class Model(BaseModel):
    """Represents an LLM with its characteristics."""

    name: str
    cost_rank: float = Field(ge=0.0, le=1.0, description="0=free, 1=most expensive")
    quality_rank: float = Field(ge=0.0, le=1.0, description="0=weakest, 1=best")
    privacy_rank: float = Field(ge=0.0, le=1.0, description="0=cloud, 1=local")
    speed_rank: float = Field(ge=0.0, le=1.0, description="0=slowest, 1=fastest")
    class_tags: str = Field(description="Space-separated tags (e.g., 'coding reasoning local')")
    endpoint: Optional[str] = None
    context_window: Optional[int] = None

    def has_tag(self, tag: str) -> bool:
        """Check if model has a specific class tag."""
        return tag.lower() in self.class_tags.lower().split()

    @property
    def tags_list(self) -> List[str]:
        """Return class tags as a list."""
        return [tag.lower() for tag in self.class_tags.split()]


class Stance(str, Enum):
    """Behavioral stance for personas."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    EXPERIMENTAL = "experimental"


class EscalationPolicy(str, Enum):
    """How to handle escalations."""

    ALWAYS = "always"
    AUTO = "auto"
    MANUAL = "manual"
    NEVER = "never"


class RiskTolerance(str, Enum):
    """Risk tolerance level."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Persona(BaseModel):
    """Represents a user persona with preferences and task specialization."""

    name: str
    preference_quality: float = Field(ge=0.0, le=1.0)
    preference_cost: float = Field(ge=0.0, le=1.0)
    preference_privacy: float = Field(ge=0.0, le=1.0)
    preference_speed: float = Field(ge=0.0, le=1.0)
    preference_creativity: float = Field(ge=0.0, le=1.0)
    stance: Stance
    escalation_policy: EscalationPolicy
    risk_tolerance: RiskTolerance
    preferred_task_types: List[str]
    boosts: Dict[str, float] = Field(default_factory=dict)

    @field_validator("boosts")
    @classmethod
    def validate_boosts(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure boost values are reasonable."""
        for key, value in v.items():
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"Boost {key} must be between -1.0 and 1.0, got {value}")
        return v


class ScoredModel(BaseModel):
    """A model with its computed score."""

    model: Model
    base_score: float
    heuristic_score: float
    total_score: float
    matched_keywords: List[str] = Field(default_factory=list)
    applied_boosts: List[str] = Field(default_factory=list)


class RankingResult(BaseModel):
    """Result of routing computation with ranked models."""

    message: str
    persona: Persona
    ranked_models: List[ScoredModel]

    @property
    def primary(self) -> ScoredModel:
        """Get the top-ranked model."""
        if not self.ranked_models:
            raise ValueError("No models available in ranking")
        return self.ranked_models[0]

    def top_n(self, n: int) -> List[ScoredModel]:
        """Get top N models."""
        return self.ranked_models[:n]


class ExecutionResult(BaseModel):
    """Result from executing a query with a model (stub for Phase 1)."""

    model_name: str
    response_text: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
