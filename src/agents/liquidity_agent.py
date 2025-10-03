"""
Liquidity & Funding Agent for RiskRadar
Analyzes LCR, NSFR, funding mix, and deposit concentrations
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class LiquidityFundingAgent(BaseAgent):
    """Agent for analyzing liquidity and funding metrics"""

    def __init__(self, model: str = None):
        super().__init__('liquidity_funding', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze liquidity and funding metrics

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing liquidity analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
