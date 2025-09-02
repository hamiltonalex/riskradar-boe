"""
Topic Evolution Agent for RiskRadar
Tracks and analyzes topic changes in earning calls
"""

from typing import Dict, List, Set
from collections import Counter
import re
from src.agents.base_agent import BaseAgent
import config

class TopicEvolutionAgent(BaseAgent):
    """Agent for tracking topic evolution and identifying emerging risks"""
    
    def __init__(self, model: str = None):
        super().__init__('topic_evolution', model)
        self.previous_topics = set()
        self.topic_history = []
        self.risk_topic_patterns = self._compile_risk_patterns()
    
    def _compile_risk_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for risk topic detection"""
        patterns = {}
        for topic in config.RISK_TOPICS:
            # Create pattern that matches the topic and related terms
            pattern_str = topic.replace(' ', r'\s+')
            patterns[topic] = re.compile(pattern_str, re.IGNORECASE)
        return patterns
    
    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze topics discussed in the transcript
        
        Args:
            text: Transcript text to analyze
            context: Optional context including previous topics
            
        Returns:
            Dict containing topic analysis results
        """
        
        # First, do a quick risk topic scan
        risk_topics_found = self._scan_for_risk_topics(text)
        
        # Prepare the prompt
        prompt = self.prompt_template + text  # No truncation - use full text
        
        # Add historical context if available
        if self.topic_history:
            recent_topics = self.topic_history[-1].get('main_topics', [])
            prompt = f"Previous call topics: {recent_topics}\n\n" + prompt
        
        # Call LLM for detailed analysis
        response = self._call_llm(prompt)
        result = self._parse_json_response(response)
        
        # Enhance results
        if result:
            # Add risk topic scan results
            result['detected_risk_topics'] = risk_topics_found
            
            # Calculate topic changes
            current_topics = set([t['topic'] for t in result.get('main_topics', [])])
            if self.previous_topics:
                result['topic_changes'] = self._analyze_topic_changes(
                    self.previous_topics, current_topics
                )
            
            # Update history
            self.previous_topics = current_topics
            self.topic_history.append(result)
            
            # Check for alerts
            result['alerts'] = self._check_topic_alerts(result)
        
        self.last_result = result
        return result
    
    def _scan_for_risk_topics(self, text: str) -> List[Dict]:
        """Scan text for predefined risk topics"""
        risk_topics = []
        
        for topic, pattern in self.risk_topic_patterns.items():
            matches = pattern.findall(text)
            if matches:
                risk_topics.append({
                    'topic': topic,
                    'frequency': len(matches),
                    'context_samples': self._extract_context(text, pattern, 2)
                })
        
        # Sort by frequency
        risk_topics.sort(key=lambda x: x['frequency'], reverse=True)
        return risk_topics
    
    def _extract_context(self, text: str, pattern: re.Pattern, num_samples: int) -> List[str]:
        """Extract context around pattern matches"""
        contexts = []
        matches = list(pattern.finditer(text))[:num_samples]
        
        for match in matches:
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()
            contexts.append(context)
        
        return contexts
    
    def _analyze_topic_changes(self, previous: Set[str], current: Set[str]) -> Dict:
        """Analyze changes between topic sets"""
        return {
            'new_topics': list(current - previous),
            'dropped_topics': list(previous - current),
            'consistent_topics': list(previous & current),
            'topic_turnover': len(current - previous) + len(previous - current)
        }
    
    def _check_topic_alerts(self, result: Dict) -> List[str]:
        """Check for conditions that should trigger alerts"""
        alerts = []
        
        # Check for multiple new risk topics
        new_risk_topics = [
            t for t in result.get('new_topics', [])
            if any(risk in t.lower() for risk in config.RISK_TOPICS)
        ]
        
        if len(new_risk_topics) >= config.ALERT_CONFIG['new_risk_topics_threshold']:
            alerts.append(f"Multiple new risk topics emerged: {', '.join(new_risk_topics)}")
        
        # Check for specific high-priority risk topics
        detected_risks = result.get('detected_risk_topics', [])
        high_priority_risks = ['commercial real estate', 'deposit outflows', 'credit risk']
        
        for risk in detected_risks:
            if risk['topic'] in high_priority_risks and risk['frequency'] > 5:
                alerts.append(f"High frequency of '{risk['topic']}' mentions: {risk['frequency']} times")
        
        # Check for strategic shifts
        if result.get('strategic_shifts'):
            alerts.append(f"Strategic shift detected: {result['strategic_shifts']}")
        
        return alerts
    
    def get_topic_trends(self) -> Dict:
        """Analyze topic trends over multiple calls"""
        if len(self.topic_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Track topic frequency over time
        topic_frequencies = {}
        
        for i, record in enumerate(self.topic_history):
            for topic_info in record.get('main_topics', []):
                topic = topic_info['topic']
                if topic not in topic_frequencies:
                    topic_frequencies[topic] = []
                topic_frequencies[topic].append({
                    'call_index': i,
                    'allocation': topic_info.get('time_allocation', '0%')
                })
        
        # Identify trending topics
        trending_up = []
        trending_down = []
        
        for topic, history in topic_frequencies.items():
            if len(history) >= 2:
                # Compare recent vs older allocations
                recent = history[-1]['allocation']
                older = history[-2]['allocation']
                
                # Convert percentage strings to floats
                recent_val = float(recent.rstrip('%')) if recent else 0
                older_val = float(older.rstrip('%')) if older else 0
                
                if recent_val > older_val * 1.5:  # 50% increase
                    trending_up.append(topic)
                elif recent_val < older_val * 0.5:  # 50% decrease
                    trending_down.append(topic)
        
        return {
            'trending_up': trending_up,
            'trending_down': trending_down,
            'total_topics_tracked': len(topic_frequencies),
            'calls_analyzed': len(self.topic_history)
        }
    
    def identify_topic_clusters(self, topics: List[str]) -> Dict[str, List[str]]:
        """Group related topics into clusters"""
        clusters = {
            'credit_and_lending': [],
            'market_and_trading': [],
            'operations_and_technology': [],
            'regulatory_and_compliance': [],
            'macroeconomic': [],
            'other': []
        }
        
        # Keywords for each cluster
        cluster_keywords = {
            'credit_and_lending': ['loan', 'credit', 'lending', 'mortgage', 'default', 'provision'],
            'market_and_trading': ['trading', 'market', 'volatility', 'investment', 'securities'],
            'operations_and_technology': ['technology', 'digital', 'cyber', 'operational', 'efficiency'],
            'regulatory_and_compliance': ['regulatory', 'compliance', 'capital', 'basel', 'requirement'],
            'macroeconomic': ['economic', 'inflation', 'rate', 'gdp', 'recession', 'growth']
        }
        
        for topic in topics:
            topic_lower = topic.lower()
            clustered = False
            
            for cluster_name, keywords in cluster_keywords.items():
                if any(keyword in topic_lower for keyword in keywords):
                    clusters[cluster_name].append(topic)
                    clustered = True
                    break
            
            if not clustered:
                clusters['other'].append(topic)
        
        # Remove empty clusters
        return {k: v for k, v in clusters.items() if v}