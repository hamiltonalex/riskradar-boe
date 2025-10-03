"""
Earnings Quality Agent for RiskRadar
Analyzes profitability metrics (ROE, ROA, NIM) and earnings sustainability
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class EarningsQualityAgent(BaseAgent):
    """Agent for analyzing earnings quality and profitability"""

    def __init__(self, model: str = None):
        super().__init__('earnings_quality', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze earnings quality metrics

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing earnings quality analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
