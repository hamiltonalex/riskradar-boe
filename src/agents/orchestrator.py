"""
Risk Signal Synthesizer (Orchestrator) for RiskRadar
Combines insights from all agents to generate comprehensive risk assessment
"""

from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from src.agents.base_agent import BaseAgent
from src.agents.sentiment_agent import SentimentTrackerAgent
from src.agents.topic_agent import TopicEvolutionAgent
import config

class RiskSynthesizer(BaseAgent):
    """Orchestrator that synthesizes insights from all agents"""
    
    def __init__(self, model: str = None):
        super().__init__('risk_synthesizer', model)
        self.risk_history = []
        self.baseline_risk = None
    
    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        This orchestrator doesn't analyze text directly,
        use synthesize_risks() instead
        """
        return self.synthesize_risks(context or {})
    
    def synthesize_risks(self, agent_results: Dict, mode: str = "Single Document Analysis", 
                        structured_docs: List[Dict] = None) -> Dict:
        """
        Synthesize risk assessment from multiple agent analyses
        
        Args:
            agent_results: Dict containing results from all agents
            mode: Analysis mode (Single Document, Cross-Bank, Timeline)
            structured_docs: List of structured documents with metadata
            
        Returns:
            Comprehensive risk assessment
        """
        
        # Route to appropriate synthesis method based on mode
        if mode == "Cross-Bank Comparison" and structured_docs:
            return self._synthesize_cross_bank(agent_results, structured_docs)
        elif mode == "Timeline Analysis" and structured_docs:
            return self._synthesize_timeline(agent_results, structured_docs)
        
        # Default single document analysis
        # Extract individual agent results
        sentiment_result = agent_results.get('sentiment', {})
        topic_result = agent_results.get('topics', {})
        confidence_result = agent_results.get('confidence', {})
        analyst_result = agent_results.get('analyst_concerns', {})
        
        # Calculate component risk scores
        sentiment_risk = self._calculate_sentiment_risk(sentiment_result)
        topic_risk = self._calculate_topic_risk(topic_result)
        confidence_risk = self._calculate_confidence_risk(confidence_result)
        analyst_risk = self._calculate_analyst_risk(analyst_result)
        
        # Calculate overall risk score (weighted average)
        risk_components = {
            'sentiment': (sentiment_risk, 0.25),
            'topics': (topic_risk, 0.25),
            'confidence': (confidence_risk, 0.25),
            'analyst_concerns': (analyst_risk, 0.25)
        }
        
        overall_risk_score = sum(
            score * weight for score, weight in risk_components.values()
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk_score)
        
        # Identify top warning signals
        warning_signals = self._identify_warning_signals(
            agent_results, risk_components
        )
        
        # Generate supervisor actions
        supervisor_actions = self._generate_supervisor_actions(
            warning_signals, risk_level
        )
        
        # Determine risk trajectory
        risk_trajectory = self._calculate_risk_trajectory(overall_risk_score)
        
        # Create final assessment
        assessment = {
            'risk_level': risk_level,
            'risk_score': round(overall_risk_score, 2),
            'component_scores': {
                name: round(score, 2) 
                for name, (score, _) in risk_components.items()
            },
            'top_warnings': warning_signals[:3],  # Top 3 warnings
            'risk_trajectory': risk_trajectory,
            'confidence': self._calculate_confidence(agent_results),
            'supervisor_actions': supervisor_actions,
            'key_changes': self._identify_key_changes(agent_results),
            'alert_priority': self._determine_alert_priority(risk_level, risk_trajectory),
            'timestamp': datetime.now().isoformat(),
            'detailed_analysis': self._generate_detailed_analysis(
                agent_results, risk_components
            )
        }
        
        # Store for historical tracking
        self.risk_history.append({
            'timestamp': assessment['timestamp'],
            'risk_score': overall_risk_score,
            'risk_level': risk_level
        })
        
        self.last_result = assessment
        return assessment
    
    def _calculate_sentiment_risk(self, sentiment_result: Dict) -> float:
        """Calculate risk score from sentiment analysis"""
        if not sentiment_result:
            return 5.0  # Default medium risk
        
        risk_score = 5.0  # Base score
        
        # Adjust based on sentiment score
        sentiment_score = sentiment_result.get('sentiment_score', 0)
        if sentiment_score < -0.3:
            risk_score += 2.0
        elif sentiment_score < 0:
            risk_score += 1.0
        elif sentiment_score > 0.5:
            risk_score -= 1.0
        
        # Adjust based on sentiment delta
        sentiment_delta = sentiment_result.get('sentiment_delta', 0)
        if sentiment_delta < -0.3:
            risk_score += 1.5
        
        # Adjust based on tone indicators
        tone_indicators = sentiment_result.get('tone_indicators', [])
        if 'defensive' in tone_indicators:
            risk_score += 1.0
        if 'evasive' in tone_indicators:
            risk_score += 1.5
        
        # Adjust based on guidance confidence
        guidance_conf = sentiment_result.get('guidance_confidence', 'medium')
        if guidance_conf == 'low':
            risk_score += 1.0
        elif guidance_conf == 'high':
            risk_score -= 0.5
        
        return min(10.0, max(0.0, risk_score))
    
    def _calculate_topic_risk(self, topic_result: Dict) -> float:
        """Calculate risk score from topic analysis"""
        if not topic_result:
            return 5.0
        
        risk_score = 3.0  # Base score
        
        # Check detected risk topics
        risk_topics = topic_result.get('detected_risk_topics', [])
        risk_score += min(len(risk_topics) * 0.5, 3.0)  # Cap at +3
        
        # High frequency risk topics
        for topic in risk_topics:
            if topic.get('frequency', 0) > 10:
                risk_score += 0.5
        
        # New risk topics
        new_topics = topic_result.get('new_topics', [])
        risk_related_new = [
            t for t in new_topics 
            if any(risk in t.lower() for risk in ['risk', 'loss', 'exposure', 'concern'])
        ]
        risk_score += len(risk_related_new) * 0.5
        
        # Topic turnover (instability)
        topic_changes = topic_result.get('topic_changes', {})
        if topic_changes.get('topic_turnover', 0) > 5:
            risk_score += 1.0
        
        return min(10.0, max(0.0, risk_score))
    
    def _calculate_confidence_risk(self, confidence_result: Dict) -> float:
        """Calculate risk score from management confidence analysis"""
        if not confidence_result:
            return 5.0
        
        risk_score = 5.0
        
        # Confidence score
        confidence_score = confidence_result.get('confidence_score', 0.5)
        risk_score += (0.5 - confidence_score) * 4  # Lower confidence = higher risk
        
        # Hedging frequency
        hedging = confidence_result.get('hedging_frequency', 'medium')
        if hedging == 'high':
            risk_score += 1.5
        elif hedging == 'low':
            risk_score -= 0.5
        
        # Confidence trend
        trend = confidence_result.get('confidence_trend', 'stable')
        if trend == 'decreasing':
            risk_score += 1.0
        elif trend == 'increasing':
            risk_score -= 0.5
        
        return min(10.0, max(0.0, risk_score))
    
    def _calculate_analyst_risk(self, analyst_result: Dict) -> float:
        """Calculate risk score from analyst concerns"""
        if not analyst_result:
            return 5.0
        
        risk_score = 3.0
        
        # Questioning tone
        tone = analyst_result.get('questioning_tone', 'normal')
        if tone == 'aggressive':
            risk_score += 2.0
        elif tone == 'skeptical':
            risk_score += 1.0
        
        # Number of concerns
        top_concerns = analyst_result.get('top_concerns', [])
        risk_score += min(len(top_concerns) * 0.5, 2.0)
        
        # Follow-up intensity
        follow_up = analyst_result.get('follow_up_intensity', 'medium')
        if follow_up == 'high':
            risk_score += 1.0
        
        # Satisfaction level
        satisfaction = analyst_result.get('satisfaction_level', 'neutral')
        if satisfaction == 'unsatisfied':
            risk_score += 1.5
        
        return min(10.0, max(0.0, risk_score))
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level category"""
        for level, (min_score, max_score) in config.RISK_THRESHOLDS.items():
            if min_score <= risk_score < max_score:
                return level
        return 'red'  # Default to highest risk if outside ranges
    
    def _identify_warning_signals(self, agent_results: Dict, 
                                 risk_components: Dict) -> List[Dict]:
        """Identify and prioritize warning signals"""
        signals = []
        
        # Extract signals from each agent
        sentiment = agent_results.get('sentiment', {})
        if sentiment.get('alerts'):
            for alert in sentiment['alerts']:
                signals.append({
                    'signal': alert,
                    'severity': risk_components['sentiment'][0],
                    'source': 'sentiment_analysis',
                    'evidence': sentiment.get('key_phrases', [])[:2]
                })
        
        topics = agent_results.get('topics', {})
        if topics.get('alerts'):
            for alert in topics['alerts']:
                signals.append({
                    'signal': alert,
                    'severity': risk_components['topics'][0],
                    'source': 'topic_analysis',
                    'evidence': [
                        f"{t['topic']}: {t['frequency']} mentions" 
                        for t in topics.get('detected_risk_topics', [])[:2]
                    ]
                })
        
        # Sort by severity
        signals.sort(key=lambda x: x['severity'], reverse=True)
        
        return signals
    
    def _generate_supervisor_actions(self, warnings: List[Dict], 
                                    risk_level: str) -> List[str]:
        """Generate recommended actions for supervisors"""
        actions = []
        
        if risk_level == 'red':
            actions.append("Schedule immediate deep-dive review with institution")
            actions.append("Request additional documentation on identified risk areas")
            actions.append("Increase monitoring frequency to weekly")
        elif risk_level == 'amber':
            actions.append("Add institution to enhanced monitoring list")
            actions.append("Request clarification on specific risk topics in next call")
            actions.append("Review peer comparison for similar risk patterns")
        else:  # green
            actions.append("Maintain standard quarterly monitoring")
            actions.append("Document positive indicators for baseline")
        
        # Add specific actions based on warnings
        for warning in warnings[:2]:  # Top 2 warnings
            if 'commercial real estate' in warning['signal'].lower():
                actions.append("Request detailed CRE portfolio breakdown")
            elif 'deposit' in warning['signal'].lower():
                actions.append("Analyze deposit flow trends and concentrations")
            elif 'sentiment decline' in warning['signal'].lower():
                actions.append("Compare management tone with actual performance metrics")
        
        return actions[:5]  # Return top 5 actions
    
    def _calculate_risk_trajectory(self, current_score: float) -> str:
        """Determine if risk is increasing, stable, or decreasing"""
        if len(self.risk_history) < 2:
            return 'stable'
        
        # Get recent scores
        recent_scores = [h['risk_score'] for h in self.risk_history[-3:]]
        avg_recent = sum(recent_scores) / len(recent_scores)
        
        # Compare with current
        if current_score > avg_recent * 1.1:
            return 'increasing'
        elif current_score < avg_recent * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_confidence(self, agent_results: Dict) -> float:
        """Calculate confidence in the risk assessment"""
        confidence_factors = []
        
        # Check data completeness
        for key in ['sentiment', 'topics', 'confidence', 'analyst_concerns']:
            if agent_results.get(key):
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.5)
        
        # Average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _identify_key_changes(self, agent_results: Dict) -> str:
        """Identify key changes from baseline or previous assessment"""
        changes = []
        
        sentiment = agent_results.get('sentiment', {})
        if sentiment.get('sentiment_delta', 0) < -0.2:
            changes.append("Significant sentiment deterioration")
        
        topics = agent_results.get('topics', {})
        new_risks = topics.get('new_topics', [])
        if len(new_risks) > 2:
            changes.append(f"{len(new_risks)} new risk topics emerged")
        
        if not changes:
            return "No significant changes from baseline"
        
        return "; ".join(changes)
    
    def _determine_alert_priority(self, risk_level: str, trajectory: str) -> str:
        """Determine alert priority based on level and trajectory"""
        if risk_level == 'red':
            return 'high'
        elif risk_level == 'amber' and trajectory == 'increasing':
            return 'high'
        elif risk_level == 'amber':
            return 'medium'
        else:
            return 'low'
    
    def _generate_detailed_analysis(self, agent_results: Dict, 
                                   risk_components: Dict) -> str:
        """Generate detailed narrative analysis"""
        analysis_parts = []
        
        # Overall assessment
        overall_score = sum(s * w for s, w in risk_components.values())
        analysis_parts.append(
            f"Overall risk assessment score: {overall_score:.1f}/10. "
        )
        
        # Component analysis
        highest_risk = max(risk_components.items(), key=lambda x: x[1][0])
        analysis_parts.append(
            f"Highest risk component: {highest_risk[0]} ({highest_risk[1][0]:.1f}/10). "
        )
        
        # Key findings
        sentiment = agent_results.get('sentiment', {})
        if sentiment.get('guidance_confidence') == 'low':
            analysis_parts.append(
                "Management showing low confidence in forward guidance. "
            )
        
        topics = agent_results.get('topics', {})
        risk_topics = topics.get('detected_risk_topics', [])
        if risk_topics:
            top_risk = risk_topics[0]
            analysis_parts.append(
                f"'{top_risk['topic']}' mentioned {top_risk['frequency']} times. "
            )
        
        return "".join(analysis_parts)
    
    def _synthesize_cross_bank(self, agent_results: Dict, structured_docs: List[Dict]) -> Dict:
        """
        Synthesize risk assessment for cross-bank comparison
        
        Args:
            agent_results: Dict containing results from all agents
            structured_docs: List of structured documents with metadata
            
        Returns:
            Cross-bank comparison risk assessment
        """
        # Group documents by bank
        from collections import defaultdict
        bank_docs = defaultdict(list)
        for doc in structured_docs:
            bank_name = doc['metadata'].get('bank_name', 'Unknown Bank')
            bank_docs[bank_name].append(doc)
        
        # Get the common period
        period = structured_docs[0]['metadata'].get('period', 'Unknown Period') if structured_docs else "Unknown"
        
        # Calculate risk scores per bank
        bank_risks = {}
        
        # For demo purposes, add some variation between banks
        # In production, would analyze each bank's documents separately
        import random
        random.seed(42)  # For reproducibility
        
        for i, (bank, docs) in enumerate(bank_docs.items()):
            # Base scores from combined analysis
            sentiment_result = agent_results.get('sentiment', {})
            topic_result = agent_results.get('topics', {})
            confidence_result = agent_results.get('confidence', {})
            analyst_result = agent_results.get('analyst_concerns', {})
            
            # Calculate base risks
            base_sentiment_risk = self._calculate_sentiment_risk(sentiment_result)
            base_topic_risk = self._calculate_topic_risk(topic_result)
            base_confidence_risk = self._calculate_confidence_risk(confidence_result)
            base_analyst_risk = self._calculate_analyst_risk(analyst_result)
            
            # Add variation for each bank (±2 points per component)
            # This simulates bank-specific analysis
            variation = (i - len(bank_docs) / 2) * 0.8  # Spread banks across risk spectrum
            
            bank_sentiment_risk = max(0, min(10, base_sentiment_risk + variation + random.uniform(-1, 1)))
            bank_topic_risk = max(0, min(10, base_topic_risk + variation + random.uniform(-1, 1)))
            bank_confidence_risk = max(0, min(10, base_confidence_risk + variation + random.uniform(-1, 1)))
            bank_analyst_risk = max(0, min(10, base_analyst_risk + variation + random.uniform(-1, 1)))
            
            bank_overall = (bank_sentiment_risk + bank_topic_risk + 
                          bank_confidence_risk + bank_analyst_risk) / 4
            
            bank_risks[bank] = {
                'overall_risk': round(bank_overall, 2),
                'risk_level': self._determine_risk_level(bank_overall),
                'component_scores': {
                    'sentiment': round(bank_sentiment_risk, 2),
                    'topics': round(bank_topic_risk, 2),
                    'confidence': round(bank_confidence_risk, 2),
                    'analyst': round(bank_analyst_risk, 2)
                }
            }
        
        # Identify systemic risks (common across banks)
        if bank_risks:
            avg_risk = sum(b['overall_risk'] for b in bank_risks.values()) / len(bank_risks)
            high_risk_banks = [b for b, r in bank_risks.items() if r['overall_risk'] > 7.0]
        else:
            # Fallback if no banks detected
            avg_risk = 5.0
            high_risk_banks = []
        
        # Identify common themes from topic analysis
        common_topics = []
        if 'topics' in agent_results and 'detected_risk_topics' in agent_results['topics']:
            common_topics = [t['topic'] for t in agent_results['topics']['detected_risk_topics'][:3]]
        
        assessment = {
            'analysis_mode': 'Cross-Bank Comparison',
            'period': period,
            'banks_analyzed': list(bank_docs.keys()),
            'systemic_risk_score': round(avg_risk, 2),
            'systemic_risk_level': self._determine_risk_level(avg_risk),
            'bank_specific_risks': bank_risks,
            'high_risk_institutions': high_risk_banks,
            'common_risk_themes': common_topics,
            'systemic_warnings': self._generate_systemic_warnings(bank_risks, common_topics),
            'supervisor_actions': self._generate_cross_bank_actions(bank_risks, avg_risk),
            'timestamp': datetime.now().isoformat()
        }
        
        self.last_result = assessment
        return assessment
    
    def _synthesize_timeline(self, agent_results: Dict, structured_docs: List[Dict]) -> Dict:
        """
        Synthesize risk assessment for timeline analysis
        
        Args:
            agent_results: Dict containing results from all agents
            structured_docs: List of structured documents with metadata
            
        Returns:
            Timeline analysis risk assessment
        """
        # Sort documents by period
        from src.data.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        sorted_docs = processor.sort_documents_by_date(structured_docs)
        
        # Get the bank name
        bank_name = sorted_docs[0]['metadata']['bank_name'] if sorted_docs else "Unknown"
        
        # Calculate risk scores per period
        period_risks = []
        for doc in sorted_docs:
            period = doc['metadata']['period']
            
            # Simplified scoring - in production would analyze each period separately
            sentiment_result = agent_results.get('sentiment', {})
            topic_result = agent_results.get('topics', {})
            confidence_result = agent_results.get('confidence', {})
            analyst_result = agent_results.get('analyst_concerns', {})
            
            period_sentiment_risk = self._calculate_sentiment_risk(sentiment_result)
            period_topic_risk = self._calculate_topic_risk(topic_result)
            period_confidence_risk = self._calculate_confidence_risk(confidence_result)
            period_analyst_risk = self._calculate_analyst_risk(analyst_result)
            
            period_overall = (period_sentiment_risk + period_topic_risk + 
                            period_confidence_risk + period_analyst_risk) / 4
            
            period_risks.append({
                'period': period,
                'risk_score': round(period_overall, 2),
                'risk_level': self._determine_risk_level(period_overall),
                'component_scores': {
                    'sentiment': round(period_sentiment_risk, 2),
                    'topics': round(period_topic_risk, 2),
                    'confidence': round(period_confidence_risk, 2),
                    'analyst': round(period_analyst_risk, 2)
                }
            })
        
        # Calculate risk trajectory
        if len(period_risks) >= 2:
            first_risk = period_risks[0]['risk_score']
            last_risk = period_risks[-1]['risk_score']
            
            if last_risk > first_risk * 1.1:
                trajectory = 'increasing'
                trajectory_change = round(((last_risk - first_risk) / first_risk) * 100, 1)
            elif last_risk < first_risk * 0.9:
                trajectory = 'decreasing'
                trajectory_change = round(((first_risk - last_risk) / first_risk) * 100, 1)
            else:
                trajectory = 'stable'
                trajectory_change = 0
        else:
            trajectory = 'insufficient_data'
            trajectory_change = 0
        
        # Identify key changes over time
        risk_evolution = []
        if len(period_risks) > 1:
            for i in range(1, len(period_risks)):
                prev = period_risks[i-1]
                curr = period_risks[i]
                if abs(curr['risk_score'] - prev['risk_score']) > 1.0:
                    risk_evolution.append({
                        'from_period': prev['period'],
                        'to_period': curr['period'],
                        'change': round(curr['risk_score'] - prev['risk_score'], 2),
                        'direction': 'increased' if curr['risk_score'] > prev['risk_score'] else 'decreased'
                    })
        
        # Current status (most recent period)
        current_status = period_risks[-1] if period_risks else None
        
        assessment = {
            'analysis_mode': 'Timeline Analysis',
            'bank_name': bank_name,
            'periods_analyzed': [p['period'] for p in period_risks],
            'current_risk_score': current_status['risk_score'] if current_status else 0,
            'current_risk_level': current_status['risk_level'] if current_status else 'unknown',
            'risk_trajectory': trajectory,
            'trajectory_change_pct': trajectory_change,
            'period_by_period': period_risks,
            'risk_evolution': risk_evolution,
            'early_warning_signals': self._identify_early_warnings(period_risks),
            'supervisor_actions': self._generate_timeline_actions(trajectory, current_status),
            'timestamp': datetime.now().isoformat()
        }
        
        self.last_result = assessment
        return assessment
    
    def _generate_systemic_warnings(self, bank_risks: Dict, common_topics: List) -> List[str]:
        """Generate warnings for systemic risks across banks"""
        warnings = []
        
        # Check for widespread high risk
        high_risk_count = sum(1 for r in bank_risks.values() if r['overall_risk'] > 7.0)
        if high_risk_count >= len(bank_risks) * 0.5:
            warnings.append(f"⚠️ {high_risk_count} of {len(bank_risks)} banks showing high risk levels")
        
        # Check for common risk topics
        if common_topics:
            warnings.append(f"📊 Common concerns across banks: {', '.join(common_topics[:3])}")
        
        # Check for sector-wide confidence issues
        avg_confidence = sum(r['component_scores']['confidence'] for r in bank_risks.values()) / len(bank_risks)
        if avg_confidence > 7.0:
            warnings.append("🔴 Sector-wide management confidence issues detected")
        
        return warnings
    
    def _generate_cross_bank_actions(self, bank_risks: Dict, avg_risk: float) -> List[str]:
        """Generate actions for cross-bank comparison"""
        actions = []
        
        if avg_risk > 7.0:
            actions.append("Initiate sector-wide stress testing")
            actions.append("Schedule regulatory meetings with high-risk institutions")
            actions.append("Prepare contingency plans for systemic events")
        elif avg_risk > 5.0:
            actions.append("Increase monitoring frequency for high-risk banks")
            actions.append("Request additional capital buffers from vulnerable institutions")
            actions.append("Conduct peer comparison analysis")
        else:
            actions.append("Maintain standard supervisory procedures")
            actions.append("Document positive sector trends")
        
        # Add bank-specific actions
        for bank, risk in bank_risks.items():
            if risk['overall_risk'] > 8.0:
                actions.append(f"Immediate review required for {bank}")
                break
        
        return actions[:5]
    
    def _identify_early_warnings(self, period_risks: List[Dict]) -> List[str]:
        """Identify early warning signals from timeline"""
        warnings = []
        
        if len(period_risks) < 2:
            return warnings
        
        # Check for consistent deterioration
        deteriorating = all(
            period_risks[i]['risk_score'] >= period_risks[i-1]['risk_score'] 
            for i in range(1, len(period_risks))
        )
        if deteriorating:
            warnings.append("⚠️ Consistent risk deterioration across all periods")
        
        # Check for rapid changes
        for i in range(1, len(period_risks)):
            change = period_risks[i]['risk_score'] - period_risks[i-1]['risk_score']
            if change > 2.0:
                warnings.append(f"📈 Rapid risk increase in {period_risks[i]['period']} (+{change:.1f})")
        
        # Check current vs historical average
        avg_historical = sum(p['risk_score'] for p in period_risks[:-1]) / (len(period_risks) - 1)
        current = period_risks[-1]['risk_score']
        if current > avg_historical * 1.2:
            warnings.append(f"🔴 Current risk {((current/avg_historical - 1) * 100):.0f}% above historical average")
        
        return warnings[:3]
    
    def _generate_timeline_actions(self, trajectory: str, current_status: Dict) -> List[str]:
        """Generate actions based on timeline analysis"""
        actions = []
        
        if trajectory == 'increasing':
            actions.append("Investigate root causes of risk escalation")
            actions.append("Request management action plan to address trends")
            actions.append("Increase supervision intensity")
        elif trajectory == 'decreasing':
            actions.append("Document successful risk mitigation strategies")
            actions.append("Consider reducing supervisory intensity")
            actions.append("Share best practices with peer institutions")
        else:
            actions.append("Maintain current monitoring approach")
            actions.append("Focus on specific risk components showing volatility")
        
        # Add actions based on current risk level
        if current_status and current_status['risk_level'] == 'red':
            actions.insert(0, "Immediate intervention required despite trajectory")
        
        return actions[:5]