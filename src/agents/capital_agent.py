"""
Capital Buffers Agent for RiskRadar
Extracts and analyzes capital adequacy metrics (CET1, Tier 1, Total Capital, Leverage)
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class CapitalBuffersAgent(BaseAgent):
    """Agent for analyzing capital adequacy and buffers"""

    def __init__(self, model: str = None):
        super().__init__('capital_buffers', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze capital metrics in the provided text

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing capital analysis results with overall_score
        """
        # Prepare the prompt using the template from config
        prompt = self.prompt_template + "\n\n" + text

        # Add context if provided
        if context:
            prompt = f"Context: {context}\n\n" + prompt

        # Call LLM for analysis
        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
