"""
Business Model Agent for RiskRadar
Analyzes revenue concentration, strategic pivots, and business sustainability
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class BusinessModelAgent(BaseAgent):
    """Agent for analyzing business model sustainability and strategic risks"""

    def __init__(self, model: str = None):
        super().__init__('business_model', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze business model sustainability

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing business model analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
