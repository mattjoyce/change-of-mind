"""Keyword-based task type matching for Phase 1 MVP."""

from typing import List, Dict, Set


class KeywordMatcher:
    """Simple keyword-based task type matching for Phase 1 MVP."""

    # Predefined keyword mappings for task types
    TASK_KEYWORDS: Dict[str, Set[str]] = {
        "coding": {
            "code",
            "script",
            "debug",
            "function",
            "class",
            "method",
            "programming",
            "python",
            "javascript",
            "bash",
            "sql",
            "bug",
            "error",
            "traceback",
            "syntax",
            "compile",
            "implement",
        },
        "reasoning": {
            "analyze",
            "compare",
            "evaluate",
            "reason",
            "logic",
            "proof",
            "theorem",
            "math",
            "calculate",
            "solve",
            "explain",
            "why",
            "how",
            "cause",
            "effect",
            "deduce",
        },
        "research": {
            "research",
            "study",
            "literature",
            "review",
            "analysis",
            "methodology",
            "hypothesis",
            "data",
            "statistics",
            "paper",
            "article",
            "journal",
            "findings",
            "investigate",
        },
        "creative": {
            "write",
            "story",
            "creative",
            "brainstorm",
            "imagine",
            "narrative",
            "character",
            "plot",
            "poetry",
            "dialogue",
            "worldbuilding",
            "fiction",
            "novel",
            "compose",
        },
        "sysadmin": {
            "server",
            "nginx",
            "apache",
            "docker",
            "kubernetes",
            "deploy",
            "infrastructure",
            "config",
            "log",
            "monitor",
            "security",
            "firewall",
            "iptables",
            "systemd",
            "cron",
        },
        "long_context": {
            "document",
            "summary",
            "summarize",
            "lengthy",
            "long",
            "multiple",
            "compare documents",
            "review all",
            "entire",
        },
    }

    def match_task_types(self, message: str, task_types: List[str]) -> List[str]:
        """
        Match message against task types using keyword detection.

        Args:
            message: User's input message
            task_types: List of task types to check (from persona.preferred_task_types)

        Returns:
            List of matched task types
        """
        message_lower = message.lower()
        matched = []

        for task_type in task_types:
            # Normalize task type (e.g., "reasoning_heavy" -> "reasoning")
            normalized = self._normalize_task_type(task_type)

            # Check if any keywords for this task type appear in message
            if normalized in self.TASK_KEYWORDS:
                keywords = self.TASK_KEYWORDS[normalized]
                if any(keyword in message_lower for keyword in keywords):
                    matched.append(task_type)

        return matched

    def detect_keywords(self, message: str) -> Dict[str, List[str]]:
        """
        Detect which keywords from each category appear in the message.

        Returns:
            Dict mapping task type to list of matched keywords
        """
        message_lower = message.lower()
        detected = {}

        for task_type, keywords in self.TASK_KEYWORDS.items():
            found = [kw for kw in keywords if kw in message_lower]
            if found:
                detected[task_type] = found

        return detected

    @staticmethod
    def _normalize_task_type(task_type: str) -> str:
        """
        Normalize task type strings.
        Examples: 'reasoning_heavy' -> 'reasoning', 'long_context' -> 'long_context'
        """
        # Extract base type (first word before underscore)
        base_types = {
            "reasoning_heavy": "reasoning",
            "literature_review": "research",
            "creative_writing": "creative",
            "config_generation": "sysadmin",
            "log_analysis": "sysadmin",
        }

        return base_types.get(task_type, task_type.split("_")[0])
