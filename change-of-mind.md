# LLM Model Router

## Overview

An intelligent routing system that selects the most appropriate Large Language Model (LLM) for each user request based on task characteristics, persona preferences, and contextual heuristics.

Rather than using a single model for all tasks, this router dynamically evaluates available models against weighted criteria (cost, quality, privacy, speed) and matches them to **persona profiles** that represent different use cases or user roles.

### Key Concept

Different tasks and users have different priorities:
- A **Researcher** prioritizes quality and reasoning over cost
- A **SysAdmin** prioritizes privacy and speed over premium features
- A **Creative Writer** prioritizes creativity and flexibility

The router makes these trade-offs explicit and automated.

---

## Problem Statement

Modern LLM ecosystems offer diverse models with different strengths:
- **Local models** (e.g., LLaMA, DeepSeek): Private, cheap, but limited quality
- **Cloud APIs** (e.g., Claude, GPT-4, Gemini): High quality, but expensive and less private
- **Specialized models**: Reasoning-focused, multimodal, long-context, coding-optimized

Users face decision fatigue choosing between them. Organizations need:
- **Cost control** without sacrificing quality where it matters
- **Privacy guarantees** for sensitive workloads
- **Optimal quality** for high-stakes tasks
- **Consistent routing logic** across teams and applications

---

## Architecture

### 1. Model Registry

Each model is characterized by weighted attributes:

```yaml
model:
  name: "Claude Sonnet 4.5"
  cost_rank: 0.4          # 0 = free, 1 = most expensive (normalized)
  quality_rank: 0.85      # 0 = weakest, 1 = best reasoning/accuracy
  privacy_rank: 0.3       # 0 = least private, 1 = fully local/private
  speed_rank: 0.7         # 0 = slowest, 1 = fastest response time
  class: "efficient instruct / conversational"
```

**Dimensions Explained:**
- `cost_rank`: Relative API cost per 1M tokens or compute requirements
- `quality_rank`: Benchmark performance (MMLU, HumanEval, reasoning tasks)
- `privacy_rank`: Data residency guarantees (1.0 = on-prem, 0.0 = third-party cloud)
- `speed_rank`: Time-to-first-token and throughput
- `class`: Free-text tags for semantic matching (e.g., "coding", "reasoning", "multimodal")

### 2. Agent Personas

Personas encode **preference profiles** and **task specialization**:

```yaml
agent_persona:
  name: "Researcher"
  
  # Core preference weights (must sum to meaningful range, typically 0-1 scale)
  preference_quality: 0.9      # strongly values accuracy and reasoning
  preference_cost: 0.1         # cost is secondary
  preference_privacy: 0.3      # cloud acceptable for non-sensitive work
  preference_speed: 0.2        # willing to wait for better results
  preference_creativity: 0.4   # moderate tolerance for creative inference
  
  # Behavioral stance
  stance: "conservative"       # Options: conservative, balanced, experimental
                               # Conservative = low hallucination tolerance, fact-checking
                               # Experimental = embrace novel approaches, rapid iteration
  
  # Task escalation rules
  escalation_policy: "always"  # Options: always, auto, manual, never
                               # "always" = escalate to better model if uncertainty detected
                               # "auto" = escalate only on model refusal or low confidence
  
  risk_tolerance: "low"        # Options: low, normal, high
                               # Affects willingness to use less-aligned or experimental models
  
  # Task specialization (used for semantic matching)
  preferred_task_types:
    - "reasoning_heavy"
    - "long_context"
    - "analysis"
    - "policy"
    - "literature_review"
  
  # Model affinity boosts (added to base score)
  boosts:
    reasoning: 0.15            # Boost models tagged with "reasoning"
    long_context: 0.10         # Boost models with >32k context windows
    high_quality: 0.10         # Boost models with quality_rank > 0.8
```

**Persona Design Principles:**
- Preferences should reflect **real user priorities**, not arbitrary weights
- Task types should be **semantically meaningful** (embeddings will match them)
- Boosts are **soft nudges**, not hard requirements (typically 0.05-0.20 range)

### 3. Routing Algorithm

The router computes a **composite score** for each model given a persona and user message:

```
score(model, persona, message) = base_score + heuristic_score

base_score = Σ (persona.preference_X × model.X_rank)
  where X ∈ {quality, cost, privacy, speed}

heuristic_score = task_match_boost + keyword_boost + boost_modifiers
```

**Base Score Calculation:**
- Multiply each persona preference by the corresponding model rank
- Invert cost (high preference_cost means prefer LOW cost_rank models)
- Normalize to 0-1 range for consistency

**Heuristic Score Components:**

1. **Task Match Boost**: 
   - Embed user message and persona's `preferred_task_types`
   - Compute cosine similarity
   - If similarity > threshold (e.g., 0.5), check if model's `class` matches task type
   - Add boost (e.g., +0.10) if aligned

2. **Keyword Boost**:
   - Simple string matching for high-confidence signals:
     - "code", "script", "debug" → boost coding-class models
     - "analyze", "research", "compare" → boost reasoning-class models
     - "image", "diagram", "visual" → boost multimodal models

3. **Boost Modifiers**:
   - Apply persona's `boosts` dictionary to matching model classes
   - Example: Researcher persona has `boosts: {reasoning: 0.15}`, so DeepSeek R1 gets +0.15

**Ranking:**
- Sort models by total score (descending)
- Return top model, or top-N candidates for fallback logic

### 4. Decision Flow

```
┌─────────────────┐
│  User Message   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Active Persona  │ ◄──── (Selected by user or context)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Compute Score for Each Model   │
│  • Base score (preferences)     │
│  • Heuristic boosts (task/kw)   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Rank Models    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Primary Model  │────►│  Execute Query   │
└─────────────────┘     └────────┬─────────┘
                                 │
                     ┌───────────┴────────────┐
                     │                        │
                     ▼                        ▼
            ┌─────────────────┐     ┌─────────────────┐
            │  Success        │     │  Failure/Low    │
            │  Return Result  │     │  Confidence     │
            └─────────────────┘     └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  Escalate to    │
                                    │  Next Best Model│
                                    └─────────────────┘
```

---

## Core Components

### 1. Model Registry (`models.yaml`)

Maintains the catalog of available models with their characteristics.

**Schema:**
```yaml
models:
  - name: string              # Human-readable model identifier
    cost_rank: float[0,1]     # Normalized cost metric
    quality_rank: float[0,1]  # Benchmark-based quality score
    privacy_rank: float[0,1]  # Data residency/privacy guarantee
    speed_rank: float[0,1]    # Response time performance
    class: string             # Space-separated tags (e.g., "coding reasoning local")
    endpoint: string          # API endpoint or local model path (optional)
    context_window: int       # Maximum context length in tokens (optional)
```

**Example:**
```yaml
models:
  - name: "LLaMA 3.1 8B (local)"
    cost_rank: 0.05
    quality_rank: 0.55
    privacy_rank: 1.0
    speed_rank: 0.9
    class: "local budget lightweight coding"
    endpoint: "ollama://llama3.1:8b"
    context_window: 8192

  - name: "DeepSeek R1"
    cost_rank: 0.4
    quality_rank: 0.9
    privacy_rank: 0.9
    speed_rank: 0.5
    class: "reasoning logic coding math"
    endpoint: "https://api.deepseek.com/v1"
    context_window: 32768
```

### 2. Persona Registry (`personas.yaml`)

Defines agent profiles with preferences and task specializations.

**Schema:**
```yaml
personas:
  - name: string
    preference_quality: float[0,1]
    preference_cost: float[0,1]
    preference_privacy: float[0,1]
    preference_speed: float[0,1]
    preference_creativity: float[0,1]
    stance: enum["conservative", "balanced", "experimental"]
    escalation_policy: enum["always", "auto", "manual", "never"]
    risk_tolerance: enum["low", "normal", "high"]
    preferred_task_types: list[string]
    boosts: dict[string, float]
```

### 3. Router Engine (`router.py`)

**Key Functions:**

```python
def route_to_model(message: str, persona: Persona, models: List[Model]) -> RankingResult:
    """
    Main routing function.
    
    Args:
        message: User's input text
        persona: Active agent persona
        models: List of available models
    
    Returns:
        RankingResult with ordered list of (model, score) tuples
    """

def compute_base_score(model: Model, persona: Persona) -> float:
    """
    Calculate weighted preference score.
    Inverts cost preference (high preference = prefer low cost).
    """

def compute_heuristic_score(message: str, model: Model, persona: Persona) -> float:
    """
    Apply task matching, keyword detection, and boost modifiers.
    Uses semantic embeddings for task-type matching.
    """

def escalate_on_failure(message: str, persona: Persona, 
                        failed_model: Model, ranked_models: List[Model]) -> Model:
    """
    Escalation handler when primary model fails or returns low confidence.
    Respects persona's escalation_policy.
    """
```

### 4. Semantic Matcher (`semantic.py`)

Handles task-type matching via embeddings.

```python
from sentence_transformers import SentenceTransformer

class TaskMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
    
    def match_task_type(self, message: str, task_types: List[str], 
                        threshold: float = 0.5) -> List[str]:
        """
        Returns task types that semantically match the message.
        
        Args:
            message: User input
            task_types: List of task type strings (e.g., ["coding", "analysis"])
            threshold: Minimum cosine similarity to consider a match
        
        Returns:
            List of matching task types
        """
```

**Why Embeddings?**
- "Help me debug this Python script" should match task_type "coding" without exact keyword
- "Compare these two research papers" should match "analysis" and "literature_review"
- More robust than keyword matching for natural language variations

### 5. Execution Layer (`executor.py`)

Handles actual model invocation and response processing.

```python
class ModelExecutor:
    def execute(self, model: Model, message: str, context: Dict) -> ExecutionResult:
        """
        Send message to selected model and return response.
        
        Returns:
            ExecutionResult containing:
                - response_text: Model's output
                - confidence: Self-reported or inferred confidence score
                - metadata: Token counts, latency, etc.
        """
    
    def check_escalation_triggers(self, result: ExecutionResult) -> bool:
        """
        Determine if response warrants escalation.
        
        Triggers:
            - Model explicitly states uncertainty ("I'm not sure...", "I don't know...")
            - Refusal to answer within scope
            - Low confidence score (if model provides one)
            - Detected hallucination markers
        """
```

---

## Configuration Examples

### Example 1: Healthcare SysAdmin Persona

```yaml
name: "Healthcare SysAdmin"
preference_quality: 0.6
preference_cost: 0.8
preference_privacy: 0.95      # HIPAA compliance critical
preference_speed: 0.7
preference_creativity: 0.2
stance: "conservative"
escalation_policy: "auto"
risk_tolerance: "low"
preferred_task_types:
  - "infrastructure"
  - "config_generation"
  - "log_analysis"
  - "security"
  - "compliance"
boosts:
  local: 0.25                 # Strong preference for on-prem models
  fast: 0.15
  deterministic: 0.10
```

**Expected Behavior:**
- Routes to local LLaMA/DeepSeek for routine sysadmin tasks
- Only escalates to cloud models when local fails or task requires advanced reasoning
- Never routes PHI-containing queries to low-privacy models

### Example 2: Research Analyst Persona

```yaml
name: "Research Analyst"
preference_quality: 0.95
preference_cost: 0.2
preference_privacy: 0.4
preference_speed: 0.3
preference_creativity: 0.5
stance: "balanced"
escalation_policy: "always"
risk_tolerance: "normal"
preferred_task_types:
  - "literature_review"
  - "data_analysis"
  - "reasoning_heavy"
  - "long_context"
  - "synthesis"
boosts:
  reasoning: 0.20
  long_context: 0.15
  high_quality: 0.10
```

**Expected Behavior:**
- Defaults to Claude Sonnet 4.5 or Gemini 2.5 Pro for most tasks
- Uses DeepSeek R1 for math-heavy or logic-intensive analysis
- Always escalates to best model if initial response seems shallow

### Example 3: Creative Writer Persona

```yaml
name: "Creative Writer"
preference_quality: 0.7
preference_cost: 0.5
preference_privacy: 0.2       # Cloud fine, content not sensitive
preference_speed: 0.6
preference_creativity: 0.9    # High tolerance for novel outputs
stance: "experimental"
escalation_policy: "manual"   # User decides when to upgrade
risk_tolerance: "high"
preferred_task_types:
  - "creative_writing"
  - "brainstorming"
  - "storytelling"
  - "dialogue"
  - "worldbuilding"
boosts:
  creative: 0.25
  conversational: 0.15
  multimodal: 0.10            # For image-based inspiration
```

**Expected Behavior:**
- Uses GPT-4o or Claude for most creative tasks
- Tolerates more "hallucination" as it's feature, not bug
- May use cheaper models for initial drafts, escalate for refinement

---

## Routing Decision Examples

### Example 1: Simple Coding Task

**Input:**
```
Persona: SysAdmin
Message: "Write a bash script to rotate nginx logs daily"
```

**Routing Logic:**
1. **Base Scores:**
   - LLaMA 3.1: 0.82 (high privacy + low cost preferred)
   - DeepSeek R1: 0.75 (slightly lower privacy, higher cost)
   - Claude Sonnet 4.5: 0.45 (low privacy, moderate cost)

2. **Heuristic Boosts:**
   - Keyword "bash script" matches task type "coding"
   - LLaMA gets +0.20 boost (local + coding class)
   - DeepSeek gets +0.15 boost (coding class)

3. **Final Scores:**
   - LLaMA 3.1: **1.02**
   - DeepSeek R1: 0.90
   - Claude Sonnet 4.5: 0.45

**Decision:** Route to LLaMA 3.1 (local, cheap, sufficient quality)

### Example 2: Complex Research Task

**Input:**
```
Persona: Researcher
Message: "Analyze the methodological differences between these three clinical trials and identify potential confounding variables"
```

**Routing Logic:**
1. **Base Scores:**
   - DeepSeek R1: 0.88 (high quality + reasoning)
   - Claude Sonnet 4.5: 0.85 (high quality)
   - Gemini 2.5 Pro: 0.87 (high quality + long context)

2. **Heuristic Boosts:**
   - Semantic match: "analysis" + "methodological" → task_type "reasoning_heavy" (similarity 0.72)
   - DeepSeek gets +0.20 boost (reasoning + high_quality boosts)
   - Gemini gets +0.15 boost (long_context + reasoning)

3. **Final Scores:**
   - DeepSeek R1: **1.08**
   - Gemini 2.5 Pro: 1.02
   - Claude Sonnet 4.5: 0.85

**Decision:** Route to DeepSeek R1, with Gemini as fallback

**Escalation Scenario:**
- If DeepSeek returns "This requires careful review of the full study protocols, which I cannot access", router escalates to Gemini 2.5 Pro (better long-context handling)

### Example 3: Ambiguous Query

**Input:**
```
Persona: SysAdmin
Message: "Help"
```

**Routing Logic:**
1. **Base Scores:**
   - LLaMA 3.1: 0.82 (persona defaults)
   - DeepSeek R1: 0.75
   - Claude Sonnet 4.5: 0.45

2. **Heuristic Boosts:**
   - No semantic match (message too short)
   - No keyword match
   - No boosts applied

3. **Final Scores:**
   - LLaMA 3.1: **0.82** (unchanged)

**Decision:** Route to LLaMA 3.1 (persona baseline)

**Follow-up Behavior:**
- If user clarifies: "Help with iptables rules", router re-scores with keyword match and may stick with LLaMA
- If user says: "Help me understand quantum entanglement", router escalates to higher-quality model

---

## Implementation Roadmap

### Phase 1: Core Router (MVP)
- [ ] Model registry (hardcoded 3-5 models)
- [ ] Persona definitions (2-3 personas: SysAdmin, Researcher, Creative)
- [ ] Base scoring function (preference weights only)
- [ ] Simple keyword matching (no embeddings yet)
- [ ] Single-model execution (no fallback)
- [ ] CLI interface for testing

**Deliverable:** Working router that selects different models for different personas

### Phase 2: Heuristics
- [ ] Semantic task matching (integrate sentence-transformers)
- [ ] Keyword boost system
- [ ] Persona boost modifiers
- [ ] Logging of routing decisions (for analysis)
- [ ] Unit tests for scoring functions

**Deliverable:** Router that intelligently adapts to message content

### Phase 3: Execution & Escalation
- [ ] Model executor with API/local model support
- [ ] Confidence detection in responses
- [ ] Automatic escalation logic
- [ ] Fallback chain (try next-best model on failure)
- [ ] Response caching (avoid re-routing identical queries)

**Deliverable:** End-to-end system with resilience

### Phase 4: Observability & Tuning
- [ ] Dashboard for routing decisions (what went where, why)
- [ ] Cost tracking per persona
- [ ] A/B testing framework (compare router vs. single-model baseline)
- [ ] Feedback loop (user thumbs-up/down adjusts weights)
- [ ] Persona recommendation engine (suggest persona based on usage patterns)

**Deliverable:** Production-ready system with continuous improvement

---

## Technical Stack

**Core Dependencies:**
- **Python 3.10+**: Core language
- **sentence-transformers**: Semantic embeddings for task matching
- **scikit-learn**: Cosine similarity calculations
- **pydantic**: Data validation for models/personas
- **PyYAML**: Configuration file parsing
- **httpx**: Async HTTP client for API calls
- **ollama-python** (optional): For local model integration

**Optional Enhancements:**
- **Redis**: Response caching
- **PostgreSQL**: Routing decision logs and analytics
- **Prometheus**: Metrics export (latency, cost per persona)
- **LangChain** (optional): If extending to multi-step agent workflows

---

## API Design

### REST API (for integration)

```http
POST /route
Content-Type: application/json

{
  "message": "Explain how transformers work in NLP",
  "persona": "Researcher",
  "context": {
    "conversation_id": "abc123",
    "previous_model": "claude-sonnet-4"
  }
}
```

**Response:**
```json
{
  "selected_model": "DeepSeek R1",
  "confidence": 0.92,
  "reasoning": {
    "base_score": 0.88,
    "heuristic_score": 0.20,
    "matched_task_types": ["reasoning_heavy"],
    "applied_boosts": ["reasoning", "high_quality"]
  },
  "alternatives": [
    {"model": "Gemini 2.5 Pro", "score": 0.87},
    {"model": "Claude Sonnet 4.5", "score": 0.75}
  ]
}
```

### Python SDK

```python
from llm_router import Router, Persona

router = Router.from_config("config.yaml")

# Simple usage
response = router.route_and_execute(
    message="Debug this Python traceback...",
    persona="SysAdmin"
)

# Advanced usage with explicit control
ranking = router.route(message, persona="Researcher")
print(f"Selected: {ranking.primary.name} (score: {ranking.primary.score})")

# Execute with fallback
result = router.execute_with_fallback(
    message, 
    ranking, 
    max_attempts=3
)
```

---

## Evaluation Metrics

### Routing Quality
- **Persona Alignment Score**: Do routing decisions match expected persona behavior?
  - Test: For 100 sample queries, human-label expected model choice, compute accuracy
- **Cost Efficiency**: Are we routing to cheaper models without sacrificing quality?
  - Metric: Average cost per query vs. "always use GPT-4" baseline
- **Task Success Rate**: Does the chosen model successfully complete the task?
  - Metric: % of queries where selected model produces satisfactory result (human eval)

### System Performance
- **Routing Latency**: Time to compute model selection
  - Target: <100ms (embeddings are the bottleneck)
- **End-to-End Latency**: Routing + model execution
  - Compare to direct API call (should be negligible overhead)
- **Escalation Rate**: % of queries that trigger fallback
  - Target: <10% (if higher, base routing needs tuning)

### Cost Analysis
- **Cost per Persona**: Track spending by persona over time
- **Cost Savings**: Compare total spend to "always use best model" baseline
- **Cost-Quality Pareto Curve**: Plot cost vs. task success rate

---

## Privacy & Security Considerations

### Data Handling
- **No Message Logging by Default**: User messages should not be persisted unless explicitly enabled
- **PII Detection**: Optional pre-router filter to detect PII/PHI and force high-privacy models
- **Audit Logs**: Log routing decisions (model chosen, scores) without message content

### Model Safety
- **Alignment Requirements**: `risk_tolerance: low` personas should never route to unaligned or jailbroken models
- **Prompt Injection Defense**: Validate that routing scores aren't manipulated by adversarial prompts in message text
- **Rate Limiting**: Prevent abuse by limiting routing requests per user/API key

### Compliance
- **HIPAA/GDPR**: Healthcare SysAdmin persona must enforce privacy_rank > 0.9 for any query containing PHI
- **Data Residency**: Models with `privacy_rank < 0.5` should not be used in EU without explicit consent

---

## Extensibility

### Custom Heuristics
Users can add domain-specific boosters:

```python
class HealthcareHeuristic(Heuristic):
    def apply(self, message: str, model: Model, persona: Persona) -> float:
        if contains_phi(message) and model.privacy_rank < 0.9:
            return -1.0  # Penalize cloud models for PHI content
        return 0.0

router.register_heuristic(HealthcareHeuristic())
```

### Multi-Model Responses
For high-stakes queries, execute multiple models and synthesize:

```python
ranking = router.route(message, persona)
responses = router.execute_parallel(ranking.top_n(3))
synthesized = router.synthesize_responses(responses)
```

### Agent Workflows
Router can integrate with agent frameworks:

```python
# LangChain integration
class RouterTool(BaseTool):
    name = "intelligent_router"
    
    def _run(self, query: str, persona: str) -> str:
        return router.route_and_execute(query, persona)
```

---

## References & Inspiration

- **Gorilla LLM**: API selection via retrieval
- **FrugalGPT**: Cost-optimized LLM cascading
- **RouteLLM**: Learning-based routing for open models
- **Semantic Router**: Embedding-based intent classification

---

## FAQ

**Q: Why not just use the best model for everything?**  
A: Cost and privacy. GPT-4 costs 30x more than LLaMA 3.1 for 90% identical results on simple tasks. And some orgs can't send data to OpenAI/Anthropic.

**Q: How does this differ from LangChain's model switching?**  
A: LangChain switches models per tool/step in a workflow. This router switches per user query based on persona preferences. They're complementary.

**Q: Can I add proprietary/fine-tuned models?**  
A: Yes. Add them to `models.yaml` with appropriate characteristics. If they're domain-specific (e.g., "MedLLaMA"), add a `domain_tags` field and create personas that boost those tags.

**Q: What if the router chooses wrong?**  
A: Escalation handles runtime failures. For systematic mis-routing, adjust persona weights or add custom heuristics. Log analysis helps identify patterns.

**Q: Does this work with multimodal models?**  
A: Yes. Add `image_support: true` to model metadata and check for image attachments in message context. Route image queries to GPT-4o/Gemini.

**Q: How do I prevent prompt injection attacks on routing?**  
A: Don't let message content directly set weights. Heuristics only add/subtract fixed amounts. Validate that scores remain in reasonable ranges.

---

## License

MIT License - see `LICENSE` file

---

## Contributing

We welcome contributions! Areas of interest:
- New heuristic strategies (e.g., using message length, user history)
- Persona templates for specific domains (legal, medical, education)
- Integration adapters for model providers (Azure OpenAI, AWS Bedrock, etc.)
- Performance benchmarks and evaluation datasets

See `CONTRIBUTING.md` for guidelines.

---

## Acknowledgments

This project synthesizes ideas from academic research on LLM routing, cost optimization in ML systems, and practical experience deploying multi-model systems in enterprise healthcare environments.

Special thanks to the open-source LLM community for making diverse, high-quality models accessible.:
