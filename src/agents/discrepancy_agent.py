"""
Discrepancy Auditor Agent for RiskRadar
Cross-checks all agent outputs for inconsistencies and missing critical disclosures
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config
import json

class DiscrepancyAuditorAgent(BaseAgent):
    """Agent for cross-validating findings and identifying discrepancies"""

    def __init__(self, model: str = None):
        super().__init__('discrepancy_auditor', model)

    def analyze(self, agent_results: Dict, context: Dict = None) -> Dict:
        """
        Cross-check agent outputs for inconsistencies

        Args:
            agent_results: Dictionary of all prior agent results
            context: Optional context

        Returns:
            Dict containing discrepancy analysis results with overall_score
        """
        # Convert agent results to JSON for the prompt
        results_summary = self._prepare_agent_results_summary(agent_results)

        # Prepare the prompt
        prompt = self.prompt_template + "\n\nAgent Results:\n" + results_summary

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        # Call LLM for analysis
        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result

    def _prepare_agent_results_summary(self, agent_results: Dict) -> str:
        """Prepare a COMPRESSED summary of agent results to avoid token limits"""
        summary = []

        # Critical quantitative agents to focus on
        critical_agents = ['capital_buffers', 'liquidity_funding', 'credit_quality', 'earnings_quality']

        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'parsed_response' in result:
                parsed = result.get('parsed_response', {})

                # COMPRESS: Extract only the most critical information for discrepancy checking
                if agent_name in critical_agents:
                    # For critical agents, extract specific metrics
                    compressed_findings = self._extract_critical_metrics(agent_name, parsed)
                else:
                    # For other agents, just extract severity and score
                    compressed_findings = {
                        'severity': parsed.get('overall_severity', 'N/A'),
                        'has_data': bool(parsed and isinstance(parsed, dict) and len(parsed) > 1)
                    }

                agent_summary = {
                    'agent': agent_name,
                    'overall_score': result.get('overall_score', 'N/A'),
                    'key_data': compressed_findings
                }
                summary.append(agent_summary)

        return json.dumps(summary, indent=2)

    def _extract_critical_metrics(self, agent_name: str, parsed_data: Dict) -> Dict:
        """Extract only critical metrics for discrepancy checking"""
        try:
            if agent_name == 'capital_buffers':
                entries = parsed_data.get('capital', {}).get('entries', [])
                return {
                    'severity': parsed_data.get('overall_severity', 'N/A'),
                    'cet1_ratio': entries[0].get('value_normalised') if entries else None,
                    'has_requirements': any(e.get('requirement_pct') for e in entries)
                }
            elif agent_name == 'liquidity_funding':
                liq = parsed_data.get('liquidity_funding', {})
                return {
                    'severity': parsed_data.get('overall_severity', 'N/A'),
                    'lcr': liq.get('lcr_pct', {}).get('value_normalised'),
                    'nsfr': liq.get('nsfr_pct', {}).get('value_normalised')
                }
            elif agent_name == 'credit_quality':
                credit = parsed_data.get('credit_quality', {})
                return {
                    'severity': parsed_data.get('overall_severity', 'N/A'),
                    'npl_ratio': credit.get('npl_metrics', {}).get('value_normalised'),
                    'stage2_present': bool(credit.get('stage_metrics'))
                }
            elif agent_name == 'earnings_quality':
                earnings = parsed_data.get('earnings_quality', {})
                return {
                    'severity': parsed_data.get('overall_severity', 'N/A'),
                    'roe': earnings.get('profitability_metrics', {}).get('ROE_pct'),
                    'cost_income': earnings.get('efficiency_metrics', {}).get('cost_to_income_pct')
                }
            else:
                return {
                    'severity': parsed_data.get('overall_severity', 'N/A'),
                    'has_data': bool(parsed_data)
                }
        except Exception:
            return {'severity': 'error', 'has_data': False}
