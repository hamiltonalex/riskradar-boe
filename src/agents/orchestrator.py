"""
Risk Signal Synthesizer (Orchestrator) for RiskRadar
Combines insights from all 16 agents to generate risk assessment
Implements 2-phase execution: 14 agents in parallel, then 2 sequential
"""

from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from src.agents.base_agent import BaseAgent
from src.agents.sentiment_agent import SentimentTrackerAgent
from src.agents.topic_agent import TopicEvolutionAgent
from src.agents.confidence_agent import ManagementConfidenceAgent
from src.agents.analyst_agent import AnalystConcernAgent
from src.agents.capital_agent import CapitalBuffersAgent
from src.agents.liquidity_agent import LiquidityFundingAgent
from src.agents.market_irrbb_agent import MarketIRRBBAgent
from src.agents.credit_agent import CreditQualityAgent
from src.agents.earnings_agent import EarningsQualityAgent
from src.agents.governance_agent import GovernanceControlsAgent
from src.agents.legal_agent import LegalRegAgent
from src.agents.business_model_agent import BusinessModelAgent
from src.agents.off_balance_sheet_agent import OffBalanceSheetAgent
from src.agents.red_flags_agent import RedFlagsAgent
from src.agents.discrepancy_agent import DiscrepancyAuditorAgent
from src.agents.camels_fusion_agent import CAMELSFuserAgent
import config

# Import logging
try:
    from src.utils.debug_logger import log_debug, log_info, log_error, log_warning
except ImportError:
    def log_debug(msg): print(f"[DEBUG] {msg}")
    def log_info(msg): print(f"[INFO] {msg}")
    def log_error(msg): print(f"[ERROR] {msg}")
    def log_warning(msg): print(f"[WARNING] {msg}")

class RiskSynthesizer(BaseAgent):
    """Orchestrator that synthesizes insights from all 16 agents with 2-phase execution"""

    def __init__(self, model: str = None):
        super().__init__('risk_synthesizer', model)
        self.risk_history = []
        self.baseline_risk = None
        self.agent_results = {}  # Store all agent results internally
        self.results_lock = threading.Lock()  # Thread safety

        # Initialize all 16 agents
        log_info(f"Initializing all 16 agents with model: {self.model}")

        # Tier 1: Linguistic agents (4)
        self.agents = {
            'sentiment': SentimentTrackerAgent(model),
            'topics': TopicEvolutionAgent(model),
            'confidence': ManagementConfidenceAgent(model),
            'analyst_concerns': AnalystConcernAgent(model),

            # Tier 2: Quantitative agents (9)
            'capital_buffers': CapitalBuffersAgent(model),
            'liquidity_funding': LiquidityFundingAgent(model),
            'market_irrbb': MarketIRRBBAgent(model),
            'credit_quality': CreditQualityAgent(model),
            'earnings_quality': EarningsQualityAgent(model),
            'governance_controls': GovernanceControlsAgent(model),
            'legal_reg': LegalRegAgent(model),
            'business_model': BusinessModelAgent(model),
            'off_balance_sheet': OffBalanceSheetAgent(model),

            # Tier 3: Pattern detection (1)
            'red_flags': RedFlagsAgent(model),

            # Tier 4: Meta-analysis (2) - run sequentially after Phase 1
            'discrepancy_auditor': DiscrepancyAuditorAgent(model),
            'camels_fuser': CAMELSFuserAgent(model)
        }

        log_info(f"Successfully initialized {len(self.agents)} agents")

    def get_agent_results(self) -> Dict:
        """Get all agent results (for external access)"""
        return self.agent_results.copy()
    
    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        This orchestrator doesn't analyze text directly,
        use synthesize_risks() instead
        """
        return self.synthesize_risks(context or {})
    
    def synthesize_risks(self, analysis_input: Dict, mode: str = "Single Document Analysis",
                        structured_docs: List[Dict] = None) -> Dict:
        """
        Synthesize risk assessment using all 16 agents with 2-phase execution

        Args:
            analysis_input: Dict with 'text' key containing document text, or pre-computed agent_results
            mode: Analysis mode (Single Document, Cross-Bank, Timeline)
            structured_docs: List of structured documents with metadata

        Returns:
            Comprehensive risk assessment from CAMELS fuser
        """

        # Check if we received pre-computed results or need to run analysis
        if 'text' in analysis_input:
            # Run full 16-agent analysis
            document_text = analysis_input['text']
            log_info("Running full 16-agent analysis...")

            # Execute all agents in 2 phases
            self._execute_all_agents(document_text)

            # Use internally stored results
            agent_results = self.agent_results
        else:
            # Use provided results (backward compatibility)
            agent_results = analysis_input

        # Route to appropriate synthesis method based on mode
        if mode == "Cross-Bank Comparison" and structured_docs:
            return self._synthesize_cross_bank(agent_results, structured_docs)
        elif mode == "Timeline Analysis" and structured_docs:
            return self._synthesize_timeline(agent_results, structured_docs)

        # Return CAMELS fuser result as final assessment
        camels_result = agent_results.get('camels_fuser', {})
        if camels_result and 'parsed_response' in camels_result:
            # Extract the final assessment from CAMELS fuser
            final_assessment = camels_result['parsed_response']

            # COMPATIBILITY LAYER: Convert overall_score to legacy format for UI
            overall_score = final_assessment.get('overall_score', 0.5)

            # Convert 0.0-1.0 scale to 0-10 scale
            risk_score_10 = overall_score * 10.0

            # Determine risk_level based on thresholds (matching config.RISK_THRESHOLDS)
            if risk_score_10 < 3.5:
                risk_level = 'green'
            elif risk_score_10 < 7.0:
                risk_level = 'amber'
            else:
                risk_level = 'red'

            # Add legacy fields for backward compatibility with UI
            final_assessment['risk_score'] = risk_score_10
            final_assessment['risk_level'] = risk_level

            # Add metadata
            final_assessment['timestamp'] = datetime.now().isoformat()
            final_assessment['agents_executed'] = len(agent_results)

            log_info(f"CAMELS assessment: {risk_level.upper()} (score: {risk_score_10:.1f}/10, overall_score: {overall_score:.3f})")

            self.last_result = final_assessment
            return final_assessment

        # FALLBACK: If CAMELS fuser didn't run, use legacy synthesis
        log_warning("CAMELS fuser not available, using legacy synthesis")

        # Extract individual agent results (legacy mode for backward compatibility)
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

    def _execute_all_agents(self, document_text: str, max_workers: int = 4):
        """
        Execute all 16 agents in 2 phases:
        Phase 1: 14 independent agents in parallel
        Phase 2: 2 dependent agents sequentially (discrepancy_auditor, camels_fuser)

        Args:
            document_text: Full document text to analyze
            max_workers: Number of concurrent threads for Phase 1
        """
        log_info("Starting 2-phase agent execution...")

        # Phase 1: Independent agents (run in parallel)
        phase1_agents = [
            'sentiment', 'topics', 'confidence', 'analyst_concerns',  # Tier 1: Linguistic
            'capital_buffers', 'liquidity_funding', 'market_irrbb',  # Tier 2: Quantitative
            'credit_quality', 'earnings_quality', 'governance_controls',
            'legal_reg', 'business_model', 'off_balance_sheet',
            'red_flags'  # Tier 3: Pattern detection
        ]

        log_info(f"Phase 1: Executing {len(phase1_agents)} agents in parallel (max_workers={max_workers})...")

        # Execute Phase 1 agents in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all Phase 1 agents
            future_to_agent = {
                executor.submit(self._run_single_agent, agent_key, document_text): agent_key
                for agent_key in phase1_agents
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_agent):
                agent_key = future_to_agent[future]
                try:
                    result = future.result()
                    completed += 1
                    log_info(f"Phase 1: {agent_key} completed ({completed}/{len(phase1_agents)})")
                except Exception as e:
                    log_error(f"Phase 1: {agent_key} failed: {str(e)}")

        log_info(f"Phase 1 complete: {completed}/{len(phase1_agents)} agents succeeded")

        # Phase 2: Dependent agents (run sequentially)
        log_info("Phase 2: Running dependent agents sequentially...")

        # Run discrepancy_auditor (needs all Phase 1 results)
        try:
            log_info("Running discrepancy_auditor...")
            discrepancy_result = self.agents['discrepancy_auditor'].analyze(self.agent_results)
            with self.results_lock:
                self.agent_results['discrepancy_auditor'] = {
                    'success': True,
                    'overall_score': discrepancy_result.get('overall_score', 0.0),
                    'parsed_response': discrepancy_result
                }
            log_info("discrepancy_auditor completed")
        except Exception as e:
            log_error(f"discrepancy_auditor failed: {str(e)}")

        # Add delay to avoid rate limits
        time.sleep(2)

        # Run camels_fuser (needs all 15 prior agents)
        try:
            log_info("Running camels_fuser...")
            camels_result = self.agents['camels_fuser'].analyze(self.agent_results)
            with self.results_lock:
                self.agent_results['camels_fuser'] = {
                    'success': True,
                    'overall_score': camels_result.get('overall_score', 0.0),
                    'parsed_response': camels_result
                }
            log_info("camels_fuser completed")
        except Exception as e:
            log_error(f"camels_fuser failed: {str(e)}")

        log_info(f"All agents complete. Total: {len(self.agent_results)}/16 agents succeeded")

    def _run_single_agent(self, agent_key: str, document_text: str) -> Dict:
        """
        Run a single agent and store results (thread-safe)

        Args:
            agent_key: Agent identifier
            document_text: Document text to analyze

        Returns:
            Agent result dictionary
        """
        try:
            start_time = time.time()
            agent = self.agents[agent_key]

            # Call the agent's analyze method
            result = agent.analyze(document_text)

            duration = time.time() - start_time

            # Store result with thread safety
            with self.results_lock:
                self.agent_results[agent_key] = {
                    'success': True,
                    'overall_score': result.get('overall_score', 0.0),
                    'parsed_response': result,
                    'duration_seconds': duration
                }

            return result
        except Exception as e:
            log_error(f"Agent {agent_key} failed: {str(e)}")
            # Store error result
            with self.results_lock:
                self.agent_results[agent_key] = {
                    'success': False,
                    'error': str(e),
                    'overall_score': 0.0
                }
            raise

    def _calculate_sentiment_risk(self, sentiment_result: Dict) -> float:
        """Calculate risk score from sentiment analysis"""
        if not sentiment_result:
            return 5.0  # Default medium risk

        # Use overall_score if available (0-1 scale from LLM)
        if 'overall_score' in sentiment_result:
            overall_score = sentiment_result.get('overall_score', 0.5)
            # Scale 0.0-1.0 to 0.0-10.0 (higher score = higher risk)
            risk_score = overall_score * 10.0
            try:
                from src.utils.debug_logger import log_debug
                log_debug(f"Sentiment: Using agent-provided overall_score: {risk_score:.2f}/10")
            except:
                pass
            return min(10.0, max(0.0, risk_score))

        # FALLBACK: Existing manual calculation for backward compatibility
        try:
            from src.utils.debug_logger import log_debug
            log_debug("Sentiment: Using manual risk calculation (legacy mode)")
        except:
            pass

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

        # Use overall_score if available (0-1 scale from LLM)
        if 'overall_score' in topic_result:
            overall_score = topic_result.get('overall_score', 0.5)
            # Scale 0.0-1.0 to 0.0-10.0 (higher score = higher risk)
            risk_score = overall_score * 10.0
            try:
                from src.utils.debug_logger import log_debug
                log_debug(f"Topic: Using agent-provided overall_score: {risk_score:.2f}/10")
            except:
                pass
            return min(10.0, max(0.0, risk_score))

        risk_score = 3.0  # Base score

        # Check detected risk topics
        risk_topics = topic_result.get('detected_risk_topics', [])
        risk_score += min(len(risk_topics) * 0.5, 3.0)  # Cap at +3

        # High frequency risk topics
        for topic in risk_topics:
            if topic.get('frequency', 0) > 10:
                risk_score += 0.5

        # Risk topics
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

        # Use overall_score if available (0-1 scale from LLM)
        if 'overall_score' in confidence_result:
            overall_score = confidence_result.get('overall_score', 0.5)
            # Scale 0.0-1.0 to 0.0-10.0 (higher score = higher risk)
            risk_score = overall_score * 10.0
            try:
                from src.utils.debug_logger import log_debug
                log_debug(f"Confidence: Using agent-provided overall_score: {risk_score:.2f}/10")
            except:
                pass
            return min(10.0, max(0.0, risk_score))

        # FALLBACK: Existing manual calculation for backward compatibility
        try:
            from src.utils.debug_logger import log_debug
            log_debug("Confidence: Using manual risk calculation (legacy mode)")
        except:
            pass

        # Get confidence score - handle both field names for compatibility
        # Prefer overall_confidence_score (0-10 scale) over confidence_score (0-1 scale)
        confidence_score = confidence_result.get('overall_confidence_score')
        if confidence_score is None:
            # Fall back to old field name and scale up
            old_score = confidence_result.get('confidence_score', 0.5)
            confidence_score = old_score * 10  # Convert 0-1 to 0-10 scale

        # Calculate risk score (10 - confidence gives us the risk)
        risk_score = 10 - confidence_score  # Direct inversion: high confidence = low risk

        # Adjust for hedging frequency
        hedging = confidence_result.get('hedging_analysis', {}).get('hedging_level',
                                      confidence_result.get('hedging_frequency', 'medium'))
        if hedging in ['high', 'significant', 'excessive']:
            risk_score = min(10.0, risk_score + 1.5)
        elif hedging in ['low', 'minimal']:
            risk_score = max(0.0, risk_score - 0.5)

        # Adjust for confidence trend
        trend = confidence_result.get('confidence_trend', 'stable')
        if trend in ['decreasing', 'deteriorating']:
            risk_score = min(10.0, risk_score + 1.0)
        elif trend in ['increasing', 'improving']:
            risk_score = max(0.0, risk_score - 0.5)

        return min(10.0, max(0.0, risk_score))
    
    def _calculate_analyst_risk(self, analyst_result: Dict) -> float:
        """Calculate risk score from analyst concerns"""
        if not analyst_result:
            return 5.0

        # Use overall_score if available (0-1 scale from LLM)
        if 'overall_score' in analyst_result:
            overall_score = analyst_result.get('overall_score', 0.5)
            # Scale 0.0-1.0 to 0.0-10.0 (higher score = higher risk)
            risk_score = overall_score * 10.0
            try:
                from src.utils.debug_logger import log_debug
                log_debug(f"Analyst: Using agent-provided overall_score: {risk_score:.2f}/10")
            except:
                pass
            return min(10.0, max(0.0, risk_score))

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
                # Handle both old (string array) and new (object array) key_phrases format
                phrases = sentiment.get('key_phrases', [])
                evidence = []
                for p in phrases[:2]:
                    if isinstance(p, dict):
                        # Format: object with 'phrase' field
                        evidence.append(p.get('phrase', ''))
                    else:
                        # Old format: string
                        evidence.append(str(p))

                signals.append({
                    'signal': alert,
                    'severity': risk_components['sentiment'][0],
                    'source': 'sentiment_analysis',
                    'evidence': evidence
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