"""
Analyst Concern Agent for RiskRadar
Analyzes analyst questions and concerns in earning call Q&A sections
"""

from typing import Dict, List, Optional, Tuple
import re
from collections import Counter, defaultdict
from src.agents.base_agent import BaseAgent
import config

class AnalystConcernAgent(BaseAgent):
    """Agent for analyzing analyst questions and concerns in Q&A sections"""
    
    def __init__(self, model: str = None):
        super().__init__('analyst_concern', model)
        
        # Concern categories and keywords
        self.concern_categories = {
            'liquidity': [
                'cash', 'liquidity', 'funding', 'deposits', 'withdrawals',
                'outflows', 'inflows', 'cash flow', 'working capital', 'covenant'
            ],
            'capital': [
                'capital', 'tier 1', 'leverage', 'ratio', 'basel',
                'regulatory capital', 'buffer', 'adequacy', 'requirements', 'RWA'
            ],
            'credit': [
                'credit', 'loan', 'defaults', 'provisions', 'NPL', 'non-performing',
                'charge-offs', 'delinquency', 'underwriting', 'portfolio quality'
            ],
            'growth': [
                'growth', 'revenue', 'margins', 'profitability', 'earnings',
                'guidance', 'outlook', 'forecast', 'targets', 'momentum'
            ],
            'risk': [
                'risk', 'exposure', 'concentration', 'hedging', 'derivatives',
                'market risk', 'operational risk', 'compliance', 'regulatory', 'litigation'
            ],
            'strategy': [
                'strategy', 'acquisition', 'merger', 'restructuring', 'cost',
                'efficiency', 'transformation', 'digital', 'competition', 'market share'
            ],
            'macro': [
                'economy', 'recession', 'inflation', 'rates', 'interest',
                'macro', 'GDP', 'unemployment', 'consumer', 'demand'
            ]
        }
        
        # Question intensity indicators
        self.intensity_markers = {
            'aggressive': [
                'why', 'how come', 'explain', 'concerned', 'worried',
                'disappointing', 'failed', 'missed', 'below', 'worse'
            ],
            'skeptical': [
                'really', 'actually', 'truly', 'sure', 'confident',
                'believe', 'seems', 'appears', 'claim', 'supposedly'
            ],
            'pressing': [
                'specifically', 'exactly', 'precisely', 'detail', 'breakdown',
                'clarify', 'elaborate', 'expand', 'follow up', 'drill down'
            ],
            'comparative': [
                'compared', 'versus', 'peers', 'competitors', 'benchmark',
                'underperform', 'outperform', 'relative', 'why not', 'others'
            ]
        }
        
        self.concern_history = []
        self.intensity_trend = []
    
    def analyze(self, text: str, context: Dict = None) -> Dict:
        """
        Analyze analyst questions and concerns in Q&A section
        
        Args:
            text: Transcript text to analyze (ideally Q&A section)
            context: Optional context including section markers
            
        Returns:
            Dict containing analyst concern analysis
        """
        
        # Extract Q&A section if full transcript provided
        qa_text = self._extract_qa_section(text)
        if not qa_text:
            qa_text = text  # Use full text if Q&A extraction fails
        
        # Parse questions and answers
        questions = self._extract_questions(qa_text)
        
        # Analyze questions for concerns
        concern_analysis = self._analyze_concerns(questions)
        question_intensity = self._analyze_question_intensity(questions)
        
        # Prepare prompt for LLM analysis
        prompt = self._create_analyst_prompt(qa_text)  # No truncation - use full Q&A text
        
        # Call LLM for detailed analysis
        response = self._call_llm(prompt)
        llm_result = self._parse_json_response(response)
        
        # Combine analyses
        result = self._combine_analyst_results(
            questions, concern_analysis, question_intensity, llm_result
        )
        
        # Calculate overall concern score
        result['concern_score'] = self._calculate_concern_score(result)
        
        # Track trends
        self._track_concern_trends(result)
        result['concern_trend'] = self._get_concern_trend()
        
        # Identify top concerns
        result['top_concerns'] = self._identify_top_concerns(result)
        result['red_flags'] = self._identify_red_flags(result)
        
        return result
    
    def _extract_qa_section(self, text: str) -> Optional[str]:
        """Extract Q&A section from full transcript"""
        # Common Q&A section markers
        qa_markers = [
            'question-and-answer', 'q&a session', 'q and a',
            'questions and answers', 'analyst questions',
            'we will now begin', 'now take questions', 'open.*questions'
        ]
        
        text_lower = text.lower()
        
        # Find Q&A section start
        qa_start = -1
        for marker in qa_markers:
            match = re.search(marker, text_lower)
            if match:
                qa_start = match.start()
                break
        
        if qa_start == -1:
            # Try to find first question pattern
            first_q = re.search(r'(analyst|question).*?:', text_lower)
            if first_q:
                qa_start = first_q.start()
        
        if qa_start != -1:
            return text[qa_start:]
        
        return None
    
    def _extract_questions(self, text: str) -> List[Dict]:
        """Extract individual questions from Q&A text"""
        questions = []
        
        # Pattern for questions (various formats)
        patterns = [
            r'(?:analyst|question).*?:(.*?)(?:answer|response|management|ceo|cfo|\Z)',
            r'q(?:uestion)?[:\s]+(.*?)a(?:nswer)?[:\s]+',
            r'(\b(?:can|could|would|will|what|why|how|when|where|is|are|do|does).*?\?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, str) and len(match) > 20:
                    questions.append({
                        'text': match[:500],  # Limit length
                        'length': len(match),
                        'question_words': self._count_question_words(match)
                    })
        
        return questions
    
    def _analyze_concerns(self, questions: List[Dict]) -> Dict:
        """Categorize concerns from analyst questions"""
        concern_counts = defaultdict(int)
        concern_questions = defaultdict(list)
        
        for question in questions:
            q_text = question['text'].lower()
            
            for category, keywords in self.concern_categories.items():
                for keyword in keywords:
                    if keyword in q_text:
                        concern_counts[category] += 1
                        concern_questions[category].append(question['text'][:150])
                        break  # Count each question once per category
        
        # Calculate percentages
        total_questions = len(questions) if questions else 1
        concern_percentages = {
            cat: round((count / total_questions) * 100, 1)
            for cat, count in concern_counts.items()
        }
        
        return {
            'concern_counts': dict(concern_counts),
            'concern_percentages': concern_percentages,
            'concern_examples': {k: v[:3] for k, v in concern_questions.items()},  # Top 3 examples
            'dominant_concerns': sorted(concern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def _analyze_question_intensity(self, questions: List[Dict]) -> Dict:
        """Analyze the intensity and tone of analyst questions"""
        intensity_scores = defaultdict(int)
        intensity_examples = defaultdict(list)
        
        for question in questions:
            q_text = question['text'].lower()
            
            for intensity_type, markers in self.intensity_markers.items():
                for marker in markers:
                    if marker in q_text:
                        intensity_scores[intensity_type] += 1
                        if len(intensity_examples[intensity_type]) < 3:
                            intensity_examples[intensity_type].append(question['text'][:150])
        
        # Calculate overall intensity
        total_intensity = sum(intensity_scores.values())
        total_questions = len(questions) if questions else 1
        intensity_ratio = round((total_intensity / total_questions) * 100, 1)
        
        # Classify intensity level
        if intensity_ratio < 20:
            intensity_level = "routine"
        elif intensity_ratio < 40:
            intensity_level = "moderate"
        elif intensity_ratio < 60:
            intensity_level = "elevated"
        else:
            intensity_level = "aggressive"
        
        return {
            'intensity_scores': dict(intensity_scores),
            'intensity_examples': dict(intensity_examples),
            'intensity_ratio': intensity_ratio,
            'intensity_level': intensity_level,
            'total_questions': len(questions)
        }
    
    def _create_analyst_prompt(self, text: str) -> str:
        """Create prompt for LLM analyst concern analysis"""
        return f"""
        Analyze the analyst questions and management responses in this Q&A section.
        Focus on:
        1. Main concerns raised by analysts
        2. Aggressiveness of questioning
        3. Satisfaction with management answers
        4. Topics that analysts repeatedly probe
        5. Any evasiveness in management responses
        
        Provide a JSON response with:
        - primary_concerns: List of top 3 analyst concerns
        - questioning_tone: "friendly", "neutral", "skeptical", or "hostile"
        - follow_up_intensity: 0-10 scale for follow-up question intensity
        - management_evasiveness: 0-10 scale for answer evasiveness
        - repeated_topics: List of topics asked multiple times
        - red_flags: List of concerning exchanges or non-answers
        - analyst_satisfaction: 0-10 scale for apparent satisfaction
        
        Q&A Text to analyze:
        {text}
        
        Response (JSON only):
        """
    
    def _combine_analyst_results(self, questions: List, concerns: Dict, 
                                 intensity: Dict, llm: Dict) -> Dict:
        """Combine all analyst analysis results"""
        if not llm:
            llm = {}
        
        return {
            'total_questions': len(questions),
            'questions_sample': questions[:5],  # First 5 questions
            'concern_analysis': concerns,
            'intensity_analysis': intensity,
            'llm_analysis': {
                'primary_concerns': llm.get('primary_concerns', []),
                'questioning_tone': llm.get('questioning_tone', 'neutral'),
                'follow_up_intensity': llm.get('follow_up_intensity', 5),
                'management_evasiveness': llm.get('management_evasiveness', 3),
                'repeated_topics': llm.get('repeated_topics', []),
                'red_flags': llm.get('red_flags', []),
                'analyst_satisfaction': llm.get('analyst_satisfaction', 7)
            }
        }
    
    def _calculate_concern_score(self, result: Dict) -> float:
        """Calculate overall analyst concern score"""
        # Extract scores
        intensity_level = result['intensity_analysis']['intensity_level']
        intensity_score_map = {
            'routine': 2, 'moderate': 4, 'elevated': 7, 'aggressive': 9
        }
        intensity_score = intensity_score_map.get(intensity_level, 5)
        
        # LLM scores
        follow_up = result['llm_analysis'].get('follow_up_intensity', 5)
        evasiveness = result['llm_analysis'].get('management_evasiveness', 3)
        satisfaction = 10 - result['llm_analysis'].get('analyst_satisfaction', 7)
        
        # Number of concerns
        concern_count = len(result['concern_analysis']['concern_counts'])
        concern_score = min(10, concern_count * 1.5)
        
        # Red flags
        red_flag_penalty = min(3, len(result['llm_analysis'].get('red_flags', [])))
        
        # Weighted calculation
        overall = (
            intensity_score * 0.25 +
            follow_up * 0.2 +
            evasiveness * 0.2 +
            satisfaction * 0.15 +
            concern_score * 0.2
        ) + red_flag_penalty
        
        return min(10, max(0, overall))
    
    def _track_concern_trends(self, result: Dict):
        """Track concern scores over time"""
        score = result['concern_score']
        self.concern_history.append(score)
        
        if len(self.concern_history) > 10:
            self.concern_history.pop(0)
        
        # Track intensity
        intensity = result['intensity_analysis']['intensity_level']
        self.intensity_trend.append(intensity)
        if len(self.intensity_trend) > 5:
            self.intensity_trend.pop(0)
    
    def _get_concern_trend(self) -> str:
        """Determine concern trend"""
        if len(self.concern_history) < 2:
            return "insufficient_data"
        
        recent = self.concern_history[-1]
        previous = sum(self.concern_history[-3:-1]) / 2 if len(self.concern_history) >= 3 else self.concern_history[-2]
        
        if recent > previous + 1.5:
            return "escalating"
        elif recent < previous - 1.5:
            return "easing"
        else:
            return "stable"
    
    def _identify_top_concerns(self, result: Dict) -> List[Dict]:
        """Identify and prioritize top analyst concerns"""
        concerns = []
        
        # From concern analysis
        for concern, count in result['concern_analysis']['dominant_concerns'][:3]:
            concerns.append({
                'category': concern,
                'frequency': count,
                'severity': 'high' if count > 3 else 'medium'
            })
        
        # From LLM analysis
        for concern in result['llm_analysis'].get('primary_concerns', [])[:2]:
            if concern not in [c['category'] for c in concerns]:
                concerns.append({
                    'category': 'other',
                    'description': concern,
                    'severity': 'medium'
                })
        
        return concerns
    
    def _identify_red_flags(self, result: Dict) -> List[str]:
        """Identify red flags from analyst interactions"""
        red_flags = []
        
        # High intensity questioning
        if result['intensity_analysis']['intensity_level'] in ['elevated', 'aggressive']:
            red_flags.append(f"Aggressive analyst questioning detected ({result['intensity_analysis']['intensity_level']})")
        
        # High evasiveness
        if result['llm_analysis'].get('management_evasiveness', 0) > 7:
            red_flags.append("High management evasiveness in responses")
        
        # Low satisfaction
        if result['llm_analysis'].get('analyst_satisfaction', 10) < 4:
            red_flags.append("Very low analyst satisfaction with answers")
        
        # Repeated topics (sign of non-answers)
        repeated = result['llm_analysis'].get('repeated_topics', [])
        if len(repeated) > 2:
            red_flags.append(f"Analysts repeatedly asking about: {', '.join(repeated[:2])}")
        
        # Add LLM-identified red flags
        red_flags.extend(result['llm_analysis'].get('red_flags', [])[:2])
        
        return red_flags
    
    def _count_question_words(self, text: str) -> int:
        """Count question words in text"""
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 
                         'which', 'can', 'could', 'would', 'will', 'do', 'does']
        text_lower = text.lower()
        return sum(1 for word in question_words if word in text_lower.split())
    
    def get_summary(self) -> Dict:
        """Get summary of analyst concern analysis"""
        if not self.concern_history:
            return {'status': 'no_data'}
        
        return {
            'current_concern_level': self.concern_history[-1] if self.concern_history else None,
            'average_concern': sum(self.concern_history) / len(self.concern_history),
            'concern_trend': self._get_concern_trend(),
            'recent_intensity': self.intensity_trend[-1] if self.intensity_trend else None,
            'history': self.concern_history[-5:]  # Last 5 readings
        }