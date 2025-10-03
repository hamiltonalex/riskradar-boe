"""
Market & IRRBB Agent for RiskRadar
Analyzes market risk and interest rate risk in the banking book
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class MarketIRRBBAgent(BaseAgent):
    """Agent for analyzing market risk and IRRBB sensitivities"""

    def __init__(self, model: str = None):
        super().__init__('market_irrbb', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze market risk and IRRBB exposures

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing market risk analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
