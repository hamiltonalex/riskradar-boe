"""
Credit Quality Agent for RiskRadar
Analyzes NPL ratios, Stage 2/3 exposures, ECL coverage, and sector concentrations
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class CreditQualityAgent(BaseAgent):
    """Agent for analyzing credit quality and asset quality metrics"""

    def __init__(self, model: str = None):
        super().__init__('credit_quality', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze credit quality metrics

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing credit quality analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
