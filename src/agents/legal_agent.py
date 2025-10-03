"""
Legal & Regulatory Agent for RiskRadar
Analyzes enforcement actions, litigation, and regulatory breaches
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class LegalRegAgent(BaseAgent):
    """Agent for analyzing legal and regulatory compliance"""

    def __init__(self, model: str = None):
        super().__init__('legal_reg', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze legal and regulatory risk

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing legal/regulatory analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
