"""
Management Confidence Agent for RiskRadar
Detects hedging, uncertainty, and confidence levels in management language
"""

from typing import Dict, List, Optional, Tuple
import re
from collections import Counter
from src.agents.base_agent import BaseAgent
import config

class ManagementConfidenceAgent(BaseAgent):
    """Agent for analyzing management confidence and hedging language"""
    
    def __init__(self, model: str = None):
        super().__init__('management_confidence', model)
        
        # Hedging and uncertainty indicators
        self.hedging_words = {
            'high_hedge': [
                'may', 'might', 'could', 'possibly', 'potentially',
                'perhaps', 'maybe', 'probable', 'likely', 'uncertain'
            ],
            'medium_hedge': [
                'believe', 'think', 'expect', 'anticipate', 'hope',
                'assume', 'suppose', 'estimate', 'project', 'forecast'
            ],
            'qualifying': [
                'however', 'although', 'despite', 'nevertheless', 'nonetheless',
                'but', 'yet', 'still', 'while', 'whereas'
            ],
            'cautious': [
                'challenges', 'concerns', 'risks', 'difficulties', 'headwinds',
                'pressures', 'obstacles', 'issues', 'problems', 'volatility'
            ]
        }
        
        # Confidence indicators
        self.confidence_words = {
            'strong': [
                'definitely', 'certainly', 'absolutely', 'clearly', 'obviously',
                'undoubtedly', 'confident', 'strong', 'robust', 'solid'
            ],
            'commitment': [
                'will', 'shall', 'committed', 'determined', 'focused',
                'dedicated', 'ensure', 'guarantee', 'promise', 'deliver'
            ],
            'positive': [
                'excellent', 'outstanding', 'exceptional', 'superior', 'great',
                'impressive', 'remarkable', 'successful', 'achieved', 'exceeded'
            ]
        }
        
        self.baseline_confidence = None
        self.confidence_history = []
    
    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze management confidence and hedging in the provided text
        
        Args:
            text: Transcript text to analyze
            context: Optional context including speaker role, section type
            
        Returns:
            Dict containing confidence analysis results
        """
        
        # First, perform rule-based analysis
        hedging_analysis = self._analyze_hedging(text)
        confidence_scores = self._calculate_confidence_scores(text)
        
        # Prepare the prompt for LLM analysis
        prompt = self._create_confidence_prompt(text)  # No truncation - use full text
        
        # Add context if provided
        if context:
            prompt = f"Context: Speaker: {context.get('speaker', 'unknown')}, " \
                    f"Role: {context.get('role', 'unknown')}, " \
                    f"Section: {context.get('section', 'unknown')}\n\n" + prompt
        
        # Call LLM for detailed analysis
        response = self._call_llm(prompt)
        llm_result = self._parse_json_response(response)
        
        # Combine rule-based and LLM results
        result = self._combine_results(hedging_analysis, confidence_scores, llm_result)
        
        # Calculate overall confidence score
        result['overall_confidence_score'] = self._calculate_overall_score(result)
        
        # Track confidence trends
        self._track_confidence_trend(result['overall_confidence_score'])
        result['confidence_trend'] = self._get_confidence_trend()
        
        # Identify specific concerns
        result['key_hedging_phrases'] = self._extract_hedging_phrases(text)
        result['confidence_indicators'] = self._extract_confidence_indicators(text)
        
        return result
    
    def _analyze_hedging(self, text: str) -> Dict:
        """Analyze hedging language in text"""
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        hedging_counts = {}
        for category, word_list in self.hedging_words.items():
            count = sum(words.count(word) for word in word_list)
            hedging_counts[category] = {
                'count': count,
                'percentage': round((count / total_words) * 100, 2) if total_words > 0 else 0
            }
        
        # Calculate hedging intensity
        total_hedging = sum(counts['count'] for counts in hedging_counts.values())
        hedging_intensity = min(10, (total_hedging / total_words) * 100) if total_words > 0 else 0
        
        return {
            'hedging_counts': hedging_counts,
            'total_hedging_words': total_hedging,
            'hedging_intensity': round(hedging_intensity, 2),
            'hedging_level': self._classify_hedging_level(hedging_intensity)
        }
    
    def _calculate_confidence_scores(self, text: str) -> Dict:
        """Calculate confidence scores based on positive indicators"""
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        confidence_counts = {}
        for category, word_list in self.confidence_words.items():
            count = sum(words.count(word) for word in word_list)
            confidence_counts[category] = {
                'count': count,
                'percentage': round((count / total_words) * 100, 2) if total_words > 0 else 0
            }
        
        # Calculate confidence strength
        total_confidence = sum(counts['count'] for counts in confidence_counts.values())
        confidence_strength = min(10, (total_confidence / total_words) * 100) if total_words > 0 else 0
        
        return {
            'confidence_counts': confidence_counts,
            'total_confidence_words': total_confidence,
            'confidence_strength': round(confidence_strength, 2),
            'confidence_level': self._classify_confidence_level(confidence_strength)
        }
    
    def _create_confidence_prompt(self, text: str) -> str:
        """Create prompt for LLM confidence analysis"""
        return f"""
        Analyze the management confidence level in this earning call transcript excerpt.
        Focus on:
        1. Language certainty vs uncertainty
        2. Use of hedging words and qualifying statements
        3. Commitment level in forward guidance
        4. Tone when discussing challenges
        5. Clarity vs vagueness in responses
        
        Provide a JSON response with:
        - confidence_score: 0-10 scale (10 = very confident)
        - hedging_examples: List of specific hedging phrases found
        - confident_statements: List of confident statements
        - guidance_clarity: Score 0-10 for guidance clarity
        - evasiveness_detected: Boolean
        - key_concerns: List of concerns expressed with uncertainty
        
        Text to analyze:
        {text}
        
        Response (JSON only):
        """
    
    def _combine_results(self, hedging: Dict, confidence: Dict, llm: Dict) -> Dict:
        """Combine rule-based and LLM analysis results"""
        if not llm:
            llm = {}
        
        return {
            'hedging_analysis': hedging,
            'confidence_analysis': confidence,
            'llm_confidence_score': llm.get('confidence_score', 5),
            'hedging_examples': llm.get('hedging_examples', []),
            'confident_statements': llm.get('confident_statements', []),
            'guidance_clarity': llm.get('guidance_clarity', 5),
            'evasiveness_detected': llm.get('evasiveness_detected', False),
            'key_concerns': llm.get('key_concerns', [])
        }
    
    def _calculate_overall_score(self, result: Dict) -> float:
        """Calculate overall confidence score combining all factors"""
        # Weight different components
        hedging_score = 10 - result['hedging_analysis']['hedging_intensity']
        confidence_score = result['confidence_analysis']['confidence_strength']
        llm_score = result.get('llm_confidence_score', 5)
        clarity_score = result.get('guidance_clarity', 5)
        
        # Penalty for evasiveness
        evasiveness_penalty = 2 if result.get('evasiveness_detected', False) else 0
        
        # Weighted average
        overall = (
            hedging_score * 0.3 +
            confidence_score * 0.3 +
            llm_score * 0.25 +
            clarity_score * 0.15
        ) - evasiveness_penalty
        
        return max(0, min(10, overall))
    
    def _track_confidence_trend(self, score: float):
        """Track confidence score over time"""
        self.confidence_history.append(score)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
        
        if self.baseline_confidence is None and len(self.confidence_history) >= 3:
            self.baseline_confidence = sum(self.confidence_history[:3]) / 3
    
    def _get_confidence_trend(self) -> str:
        """Determine confidence trend"""
        if len(self.confidence_history) < 2:
            return "insufficient_data"
        
        recent = self.confidence_history[-1]
        previous = sum(self.confidence_history[-3:-1]) / 2 if len(self.confidence_history) >= 3 else self.confidence_history[-2]
        
        if recent > previous + 1:
            return "improving"
        elif recent < previous - 1:
            return "deteriorating"
        else:
            return "stable"
    
    def _classify_hedging_level(self, intensity: float) -> str:
        """Classify hedging intensity level"""
        if intensity < 2:
            return "minimal"
        elif intensity < 4:
            return "moderate"
        elif intensity < 6:
            return "significant"
        else:
            return "excessive"
    
    def _classify_confidence_level(self, strength: float) -> str:
        """Classify confidence strength level"""
        if strength < 2:
            return "very_low"
        elif strength < 4:
            return "low"
        elif strength < 6:
            return "moderate"
        elif strength < 8:
            return "high"
        else:
            return "very_high"
    
    def _extract_hedging_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """Extract specific hedging phrases from text"""
        phrases = []
        sentences = text.split('.')
        
        for sentence in sentences[:50]:  # Check first 50 sentences
            sentence_lower = sentence.lower()
            for word in self.hedging_words['high_hedge']:
                if word in sentence_lower:
                    # Extract surrounding context
                    phrase = sentence.strip()[:150]
                    if phrase and phrase not in phrases:
                        phrases.append(phrase)
                        if len(phrases) >= max_phrases:
                            return phrases
        
        return phrases
    
    def _extract_confidence_indicators(self, text: str, max_indicators: int = 5) -> List[str]:
        """Extract confident statements from text"""
        indicators = []
        sentences = text.split('.')
        
        for sentence in sentences[:50]:  # Check first 50 sentences
            sentence_lower = sentence.lower()
            for word in self.confidence_words['strong'] + self.confidence_words['commitment']:
                if word in sentence_lower:
                    # Extract the statement
                    statement = sentence.strip()[:150]
                    if statement and statement not in indicators:
                        indicators.append(statement)
                        if len(indicators) >= max_indicators:
                            return indicators
        
        return indicators
    
    def get_summary(self) -> Dict:
        """Get summary of confidence analysis"""
        if not self.confidence_history:
            return {'status': 'no_data'}
        
        return {
            'current_confidence': self.confidence_history[-1] if self.confidence_history else None,
            'average_confidence': sum(self.confidence_history) / len(self.confidence_history),
            'confidence_trend': self._get_confidence_trend(),
            'baseline': self.baseline_confidence,
            'history': self.confidence_history[-5:]  # Last 5 readings
        }