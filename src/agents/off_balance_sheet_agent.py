"""
Off-Balance Sheet Agent for RiskRadar
Analyzes commitments, guarantees, derivatives, and SPV exposures
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class OffBalanceSheetAgent(BaseAgent):
    """Agent for analyzing off-balance sheet exposures and hidden risks"""

    def __init__(self, model: str = None):
        super().__init__('off_balance_sheet', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze off-balance sheet exposures

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing off-balance sheet analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
