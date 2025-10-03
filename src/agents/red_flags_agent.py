"""
Red Flags Agent for RiskRadar
Scans for critical warning signals (going concern, covenant breaches, material weaknesses)
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class RedFlagsAgent(BaseAgent):
    """Agent for detecting critical warning signals and red flags"""

    def __init__(self, model: str = None):
        super().__init__('red_flags', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Scan for critical warning signals

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing red flags analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
