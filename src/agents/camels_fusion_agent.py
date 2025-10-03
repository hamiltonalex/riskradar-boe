"""
CAMELS Fuser Agent for RiskRadar
Produces final comprehensive risk report with CAMELS ratings
"""

from typing import Dict
from src.agents.base_agent import BaseAgent
import config
import json

class CAMELSFuserAgent(BaseAgent):
    """Agent for synthesizing all findings into final CAMELS assessment"""

    def __init__(self, model: str = None):
        super().__init__('camels_fuser', model)

    def analyze(self, agent_results: Dict, context: Dict = None) -> Dict:
        """
        Synthesize all agent results into CAMELS assessment

        Args:
            agent_results: Dictionary of all prior agent results (including discrepancy_auditor)
            context: Optional context

        Returns:
            Dict containing final CAMELS assessment with overall_score
        """
        # Convert agent results to JSON for the prompt
        results_summary = self._prepare_comprehensive_summary(agent_results)

        # Prepare the prompt
        prompt = self.prompt_template + "\n\nAll Agent Results:\n" + results_summary

        if context:
            prompt = f"Context: {context}\n\n" + prompt

        # Call LLM for final synthesis
        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        self.last_result = result
        return result

    def _prepare_comprehensive_summary(self, agent_results: Dict) -> str:
        """Prepare COMPRESSED summary of all agent results to avoid token limits"""
        summary = []

        # Define agent categories for weighted scoring
        linguistic_agents = ['sentiment_tracker', 'topic_analyzer', 'confidence_evaluator', 'analyst_concern']
        quantitative_agents = ['capital_buffers', 'liquidity_funding', 'market_irrbb', 'credit_quality',
                             'earnings_quality', 'governance_controls', 'legal_reg', 'business_model',
                             'off_balance_sheet']
        meta_agents = ['red_flags', 'discrepancy_auditor']

        for agent_name, result in agent_results.items():
            if isinstance(result, dict) and 'parsed_response' in result:
                # Determine agent category
                if agent_name in linguistic_agents:
                    category = 'linguistic'
                    weight = 0.30
                elif agent_name in quantitative_agents:
                    category = 'quantitative'
                    weight = 0.60
                elif agent_name in meta_agents:
                    category = 'meta_analysis'
                    weight = 0.10
                else:
                    category = 'unknown'
                    weight = 0.0

                parsed = result.get('parsed_response', {})

                # COMPRESS: Extract only key fields instead of entire findings
                compressed_findings = {}

                # For each agent, extract only the most critical fields
                if agent_name == 'capital_buffers':
                    compressed_findings = {
                        'severity': parsed.get('overall_severity', 'N/A'),
                        'cet1_ratio': parsed.get('capital', {}).get('entries', [{}])[0].get('value_normalised'),
                        'key_metrics': self._extract_key_metrics(parsed)
                    }
                elif agent_name == 'liquidity_funding':
                    compressed_findings = {
                        'severity': parsed.get('overall_severity', 'N/A'),
                        'lcr': parsed.get('liquidity_funding', {}).get('lcr_pct', {}).get('value_normalised'),
                        'nsfr': parsed.get('liquidity_funding', {}).get('nsfr_pct', {}).get('value_normalised')
                    }
                elif agent_name == 'red_flags':
                    compressed_findings = {
                        'critical_count': parsed.get('flag_count_by_severity', {}).get('critical', 0),
                        'major_count': parsed.get('flag_count_by_severity', {}).get('major', 0),
                        'top_flags': [f.get('flag_type') for f in parsed.get('red_flags_detected', [])[:3]]
                    }
                elif agent_name == 'discrepancy_auditor':
                    compressed_findings = {
                        'materiality': parsed.get('materiality_assessment', 'N/A'),
                        'discrepancy_count': len(parsed.get('discrepancies', [])),
                        'missing_critical': parsed.get('missing_critical', [])[:3]  # Top 3 only
                    }
                elif agent_name in linguistic_agents:
                    # For linguistic agents, just keep summary fields
                    compressed_findings = {
                        'severity': parsed.get('overall_severity', 'N/A'),
                        'key_finding': self._get_first_key_finding(parsed),
                        'concern_level': parsed.get('concern_intensity', parsed.get('confidence_score'))
                    }
                else:
                    # For other agents, extract just severity and one key metric
                    compressed_findings = {
                        'severity': parsed.get('overall_severity', 'N/A'),
                        'summary': str(parsed)[:200] if parsed else 'N/A'  # Truncate to 200 chars
                    }

                agent_summary = {
                    'agent': agent_name,
                    'category': category,
                    'weight': weight,
                    'overall_score': result.get('overall_score', 'N/A'),
                    'findings': compressed_findings  # Use compressed findings instead of full
                }
                summary.append(agent_summary)

        return json.dumps(summary, indent=2)

    def _extract_key_metrics(self, parsed_data: Dict) -> list:
        """Extract only key metric names from parsed data"""
        try:
            entries = parsed_data.get('capital', {}).get('entries', [])
            return [e.get('metric') for e in entries[:3]]  # Top 3 metrics only
        except:
            return []

    def _get_first_key_finding(self, parsed_data: Dict) -> str:
        """Get the first significant finding from linguistic analysis"""
        try:
            # Try different fields depending on the agent
            if 'key_phrases' in parsed_data:
                phrases = parsed_data.get('key_phrases', [])
                if phrases:
                    return phrases[0].get('phrase', 'N/A')[:100]  # Truncate to 100 chars
            elif 'top_concerns' in parsed_data:
                concerns = parsed_data.get('top_concerns', [])
                if concerns:
                    return concerns[0].get('topic', 'N/A')[:100]
            elif 'emerging_topics' in parsed_data:
                topics = parsed_data.get('emerging_topics', [])
                if topics:
                    return topics[0].get('topic', 'N/A')[:100]
            return 'N/A'
        except:
            return 'N/A'
