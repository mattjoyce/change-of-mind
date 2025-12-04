# Change of Mind - LLM Model Router

An intelligent routing system that selects the most appropriate Large Language Model (LLM) for each user request based on task characteristics, persona preferences, and contextual heuristics.

## Overview

Different tasks and users have different priorities:
- A **Researcher** prioritizes quality and reasoning over cost
- A **SysAdmin** prioritizes privacy and speed over premium features
- A **Creative Writer** prioritizes creativity and flexibility
- A **General** user wants balanced routing that adapts to the task at hand

This router makes these trade-offs explicit and automated.

## Current Status: Phase 1 MVP

This is Phase 1 - a minimal viable product with:
- ✅ Model registry (5 models: local, mid-tier, premium)
- ✅ Persona definitions (4 personas: SysAdmin, Researcher, Creative, General)
- ✅ Base scoring function (preference weights)
- ✅ Simple keyword matching (no embeddings)
- ✅ Mock execution (no real API calls)
- ✅ CLI interface for testing
- ✅ Detailed explanation mode (--explain)

**What's NOT in Phase 1:**
- ❌ Semantic task matching with embeddings (Phase 2)
- ❌ Real model execution/API integration (Phase 3)
- ❌ Automatic escalation on failure (Phase 3)
- ❌ Response caching (Phase 4)

## Installation

### Prerequisites
- Python 3.10+
- Virtual environment support

### Setup

1. Navigate to the project directory:
```bash
cd /home/matt/project/change-of-mind
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

Basic usage:
```bash
python -m change_of_mind.cli "Help me write a bash script to rotate logs" --persona SysAdmin
```

With execution (mock in Phase 1):
```bash
python -m change_of_mind.cli "Analyze this research paper methodology" \
  --persona Researcher \
  --execute
```

List available personas:
```bash
python -m change_of_mind.cli --list-personas
```

List available models:
```bash
python -m change_of_mind.cli --list-models
```

Show top 5 models:
```bash
python -m change_of_mind.cli "Debug Python code" \
  --persona SysAdmin \
  --top-n 5
```

Show detailed scoring breakdown (the working out):
```bash
python -m change_of_mind.cli "Write a bash script to rotate logs" \
  --persona SysAdmin \
  --explain
```
This shows:
- Persona preferences and boosts
- Model characteristics for each model
- Step-by-step base score calculation (with cost inversion shown)
- Heuristic score breakdown with matched keywords and applied boosts
- Total score calculation for all models

### Python API

```python
from change_of_mind.config import Config
from change_of_mind.router import Router
from change_of_mind.scorer import Scorer
from change_of_mind.matcher import KeywordMatcher

# Load configuration
config = Config.from_directory("config")

# Initialize router
router = Router(config.models, Scorer(KeywordMatcher()))

# Route a message
persona = config.get_persona("Researcher")
ranking = router.route(
    "Compare these two machine learning approaches",
    persona
)

# Get result
print(f"Selected model: {ranking.primary.model.name}")
print(f"Score: {ranking.primary.total_score:.3f}")

# Get explanation
print(router.explain_decision(ranking))
```

## Personas Explained

### SysAdmin
Prioritizes **privacy** (0.9) and **cost** (0.8). Routes to local models (LLaMA) for most tasks to keep data private and costs low.

### Researcher
Prioritizes **quality** (0.95) above all else. Cost is secondary (0.2). Routes to premium models (Claude, DeepSeek R1) for analysis and reasoning.

### Creative
Prioritizes **creativity** (0.9) and **speed** (0.6). Experimental stance. Routes to models with creative tags (GPT-4o, Claude).

### General
**Balanced preferences** (0.5-0.7 range) across all dimensions. Has broad task type coverage (coding, reasoning, creative, sysadmin, research, long_context) with minimal boosts (+0.10). This allows the **heuristic scoring to dominate** - the router picks the best model based on what the task actually needs rather than persona bias.

**Design philosophy**: General persona trusts the router to make smart decisions based on keyword matching and model specializations. It's ideal for:
- Multi-purpose applications
- Users who want "smart defaults"
- Situations where task variety is high

## Configuration

Models and personas are defined in YAML files in the `config/` directory:

- `config/models.yaml` - Model registry with characteristics
- `config/personas.yaml` - Persona definitions with preferences

### Adding New Models

Edit `config/models.yaml`:
```yaml
- name: "Your Model Name"
  cost_rank: 0.5          # 0-1 scale
  quality_rank: 0.7       # 0-1 scale
  privacy_rank: 0.6       # 0-1 scale
  speed_rank: 0.8         # 0-1 scale
  class_tags: "coding reasoning"  # Space-separated tags
```

### Adding New Personas

Edit `config/personas.yaml`:
```yaml
- name: "YourPersona"
  preference_quality: 0.8
  preference_cost: 0.5
  preference_privacy: 0.7
  preference_speed: 0.6
  preference_creativity: 0.4
  stance: "balanced"
  escalation_policy: "auto"
  risk_tolerance: "normal"
  preferred_task_types:
    - "your_task_type"
  boosts:
    your_tag: 0.15
```

## Project Structure

```
change-of-mind/
├── change_of_mind/          # Main package
│   ├── __init__.py
│   ├── models.py            # Pydantic data models
│   ├── router.py            # Core routing logic
│   ├── scorer.py            # Scoring algorithms
│   ├── matcher.py           # Keyword matching
│   ├── executor.py          # Mock executor
│   ├── config.py            # Configuration loading
│   └── cli.py               # CLI interface
├── config/                  # Configuration files
│   ├── models.yaml
│   └── personas.yaml
├── tests/                   # Unit tests (optional)
├── venv/                    # Virtual environment
├── requirements.txt
└── README.md
```

## How It Works

### Routing Algorithm

The router computes a **composite score** for each model:

```
score(model, persona, message) = base_score + heuristic_score
```

**Base Score** - Static preferences:
- Multiply each persona preference by the corresponding model rank
- **Cost is inverted**: high `preference_cost` means prefer LOW `cost_rank` models
- Normalize to 0-1 range

**Heuristic Score** - Dynamic context:
1. **Task matching**: +0.10 if persona's preferred task types match message keywords
2. **Keyword boost**: +0.03 per keyword (max +0.15)
3. **Persona boosts**: Apply custom boosts from persona config
4. **Special heuristics**: High privacy personas get +0.10 for local models

The model with the highest total score is selected.

## Examples

### SysAdmin routing coding task:
```bash
python -m change_of_mind.cli "Write a bash script to rotate nginx logs" --persona SysAdmin
```
Expected: LLaMA 3.1 (local, cheap, private)

### Researcher routing analysis task:
```bash
python -m change_of_mind.cli "Analyze the methodological differences in these clinical trials" --persona Researcher
```
Expected: DeepSeek R1 or Claude (high quality, reasoning)

### Creative routing writing task:
```bash
python -m change_of_mind.cli "Write a short story about a time traveler" --persona Creative
```
Expected: GPT-4o or Claude (creative tags)

### Detailed explanation (show the working):
```bash
python -m change_of_mind.cli "Analyze methodological differences in clinical trials" --persona Researcher --explain
```
Shows complete scoring breakdown for all 5 models with step-by-step calculations

### General persona adapts to different tasks:
```bash
# Coding task → Gemini Flash (fast, cheap, good for coding)
python -m change_of_mind.cli "Write a bash script to backup databases" --persona General

# Reasoning task → DeepSeek R1 (reasoning specialist)
python -m change_of_mind.cli "Analyze the philosophical implications of AI" --persona General

# Creative task → Gemini Flash or GPT-4o (creative tags)
python -m change_of_mind.cli "Write a poem about the ocean" --persona General

# Research task → Claude Sonnet 4.5 (high quality, long context)
python -m change_of_mind.cli "Compare and summarize multiple research papers" --persona General
```

## Development

### Code Quality

Format code:
```bash
source venv/bin/activate
black change_of_mind/
```

Lint code:
```bash
pylint change_of_mind/
```

## Roadmap

- **Phase 2**: Semantic task matching with embeddings (sentence-transformers)
- **Phase 3**: Real execution with API integration and escalation
- **Phase 4**: Observability, cost tracking, and continuous improvement

## License

MIT License - see [LICENSE](LICENSE) file

## Documentation

For the full specification and design details, see [change-of-mind.md](change-of-mind.md).
