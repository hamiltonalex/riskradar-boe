"""
Governance & Controls Agent for RiskRadar
Analyzes internal controls, audit opinions, and compliance issues
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config

class GovernanceControlsAgent(BaseAgent):
    """Agent for analyzing governance and internal controls"""

    def __init__(self, model: str = None):
        super().__init__('governance_controls', model)

    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze governance and control frameworks

        Args:
            text: Document text to analyze
            context: Optional context

        Returns:
            Dict containing governance analysis results with overall_score
        """
        prompt = self.prompt_template + "\n\n" + text

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result
