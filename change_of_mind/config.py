"""Configuration loading for models and personas."""

import yaml
from pathlib import Path
from typing import List, Dict
from .models import Model, Persona


class Config:
    """Loads and manages configuration from YAML files."""

    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.models: List[Model] = []
        self.personas: Dict[str, Persona] = {}

    def load_models(self, filename: str = "models.yaml") -> List[Model]:
        """Load model registry from YAML."""
        filepath = self.config_dir / filename
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        self.models = [Model(**model_data) for model_data in data["models"]]
        return self.models

    def load_personas(self, filename: str = "personas.yaml") -> Dict[str, Persona]:
        """Load persona definitions from YAML."""
        filepath = self.config_dir / filename
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        self.personas = {p_data["name"]: Persona(**p_data) for p_data in data["personas"]}
        return self.personas

    def get_persona(self, name: str) -> Persona:
        """Get persona by name."""
        return self.personas[name]

    @classmethod
    def from_directory(cls, config_dir: str) -> "Config":
        """Factory method to load all config from a directory."""
        config = cls(config_dir)
        config.load_models()
        config.load_personas()
        return config
