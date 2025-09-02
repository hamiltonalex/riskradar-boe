"""
Sentiment Tracker Agent for RiskRadar
Analyzes management tone and sentiment in earning calls
"""

from typing import Dict, List, Optional
import re
from textblob import TextBlob
from src.agents.base_agent import BaseAgent
import config

class SentimentTrackerAgent(BaseAgent):
    """Agent for analyzing sentiment and tone in earning call transcripts"""
    
    def __init__(self, model: str = None):
        super().__init__('sentiment_tracker', model)
        self.baseline_sentiment = None
        self.historical_sentiments = []
    
    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze sentiment and tone in the provided text
        
        Args:
            text: Transcript text to analyze
            context: Optional context including speaker role, section type
            
        Returns:
            Dict containing sentiment analysis results
        """
        
        # First, get a quick sentiment score using TextBlob for validation
        blob_sentiment = self._get_textblob_sentiment(text)
        
        # Prepare the prompt
        prompt = self.prompt_template + text  # No artificial limit - let LLM handle context window
        
        # Add context if provided
        if context:
            prompt = f"Context: Speaker role: {context.get('role', 'unknown')}, " \
                    f"Section: {context.get('section', 'unknown')}\n\n" + prompt
        
        # Call LLM for detailed analysis
        response = self._call_llm(prompt)
        result = self._parse_json_response(response)
        
        # Validate and enhance results
        if result:
            # Add TextBlob sentiment for comparison
            result['textblob_sentiment'] = blob_sentiment
            
            # Calculate sentiment delta if we have historical data
            if self.historical_sentiments:
                result['sentiment_delta'] = self._calculate_sentiment_delta(
                    result.get('sentiment_score', 0)
                )
            
            # Store for historical comparison
            self.historical_sentiments.append(result.get('sentiment_score', 0))
            
            # Check for alert conditions
            result['alerts'] = self._check_sentiment_alerts(result)
        
        self.last_result = result
        return result
    
    def _get_textblob_sentiment(self, text: str) -> float:
        """Get sentiment score using TextBlob as a baseline"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _calculate_sentiment_delta(self, current_score: float) -> float:
        """Calculate change from previous sentiment"""
        if not self.historical_sentiments:
            return 0.0
        
        previous = self.historical_sentiments[-1]
        return current_score - previous
    
    def _check_sentiment_alerts(self, result: Dict) -> List[str]:
        """Check for conditions that should trigger alerts"""
        alerts = []
        
        # Check for significant sentiment drop
        if result.get('sentiment_delta', 0) < config.ALERT_CONFIG['sentiment_drop_threshold']:
            alerts.append(f"Significant sentiment decline: {result['sentiment_delta']:.2f}")
        
        # Check for defensive language
        if 'defensive' in result.get('tone_indicators', []):
            alerts.append("Defensive tone detected in management communication")
        
        # Check for low guidance confidence
        if result.get('guidance_confidence') == 'low':
            alerts.append("Low confidence in forward guidance")
        
        # Check for extreme sentiment
        sentiment_score = result.get('sentiment_score', 0)
        if sentiment_score < -0.5:
            alerts.append("Strongly negative sentiment detected")
        elif sentiment_score > 0.7:
            alerts.append("Unusually optimistic sentiment - potential overconfidence")
        
        return alerts
    
    def analyze_speaker_comparison(self, segments: List[Dict]) -> Dict:
        """Compare sentiment across different speakers"""
        speaker_sentiments = {}
        
        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('content', '')
            
            if speaker not in speaker_sentiments:
                speaker_sentiments[speaker] = []
            
            # Analyze this segment
            result = self.analyze(text, {'role': segment.get('role')})
            speaker_sentiments[speaker].append(result.get('sentiment_score', 0))
        
        # Calculate averages and identify discrepancies
        analysis = {}
        for speaker, scores in speaker_sentiments.items():
            if scores:
                analysis[speaker] = {
                    'average_sentiment': sum(scores) / len(scores),
                    'min_sentiment': min(scores),
                    'max_sentiment': max(scores),
                    'variance': self._calculate_variance(scores)
                }
        
        # Check for discrepancies between CEO and CFO
        if 'CEO' in analysis and 'CFO' in analysis:
            ceo_avg = analysis['CEO']['average_sentiment']
            cfo_avg = analysis['CFO']['average_sentiment']
            discrepancy = abs(ceo_avg - cfo_avg)
            
            if discrepancy > 0.3:
                analysis['alert'] = f"Significant sentiment discrepancy between CEO and CFO: {discrepancy:.2f}"
        
        return analysis
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of sentiment scores"""
        if not scores:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance
    
    def get_trend_analysis(self) -> Dict:
        """Analyze sentiment trends over time"""
        if len(self.historical_sentiments) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate moving average
        window_size = min(3, len(self.historical_sentiments))
        recent_avg = sum(self.historical_sentiments[-window_size:]) / window_size
        
        # Determine trend
        if len(self.historical_sentiments) >= 4:
            older_avg = sum(self.historical_sentiments[-2*window_size:-window_size]) / window_size
            trend_direction = 'improving' if recent_avg > older_avg else 'declining'
            trend_strength = abs(recent_avg - older_avg)
        else:
            trend_direction = 'stable'
            trend_strength = 0
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'recent_average': recent_avg,
            'data_points': len(self.historical_sentiments)
        }