"""
Configuration and Prompts for RiskRadar Multi-Agent System
"""

import os
from dotenv import load_dotenv

# Load environment variables (override=True to prefer .env over shell)
load_dotenv(override=True)

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Model Configuration
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-5-mini')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 35000))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))

# Agent Prompts
AGENT_PROMPTS = {
    'sentiment_tracker': """
You are a financial sentiment analysis expert. Analyze the provided earning call transcript segment 
and evaluate the sentiment and tone of management communication.

Analyze sentiment concisely focusing on:
1. Overall sentiment with score (-1 to 1)
2. Key tone indicators (max 3)
3. Most important phrases (max 3)

Output format (JSON):
{
    "overall_sentiment": "positive/negative/neutral",
    "sentiment_score": -1.0 to 1.0,
    "confidence_level": 0.0 to 1.0,
    "tone_indicators": ["defensive", "optimistic", etc.],
    "key_phrases": ["specific quotes showing sentiment"],
    "guidance_confidence": "high/medium/low",
    "notable_changes": "description of tone shifts"
}

Transcript segment:
""",

    'topic_evolution': """
You are a financial discourse analyst. Analyze the earning call transcript to identify 
and track discussion topics and their evolution.

Identify main topics concisely:
1. Top 3-5 main topics with rough percentages
2. Any new risk topics (max 2)
3. Key strategic shifts (if any)

Output format (JSON):
{
    "main_topics": [
        {"topic": "name", "time_allocation": "percentage", "sentiment": "positive/negative/neutral"}
    ],
    "new_topics": ["topic1", "topic2"],
    "increased_focus": ["topic1", "topic2"],
    "decreased_focus": ["topic1", "topic2"],
    "risk_topics": ["risk1", "risk2"],
    "strategic_shifts": "description of priority changes"
}

Transcript segment:
""",

    'management_confidence': """
You are a linguistic analyst specializing in executive communication. Analyze management's 
language to assess confidence levels and certainty.

Focus on:
1. Hedging language ("might", "could", "possibly")
2. Certainty indicators ("will", "definitely", "committed")
3. Qualifying statements and caveats
4. Comparison to previous quarter's language strength
5. Guidance specificity level

Output format (JSON):
{
    "confidence_score": 0.0 to 1.0,
    "hedging_frequency": "high/medium/low",
    "hedging_examples": ["specific quotes"],
    "certainty_phrases": ["specific quotes"],
    "qualifier_count": integer,
    "guidance_specificity": "specific/vague",
    "confidence_trend": "increasing/stable/decreasing",
    "areas_of_uncertainty": ["topic1", "topic2"]
}

Transcript segment:
""",

    'analyst_concern': """
You are an expert at analyzing sell-side analyst behavior and concerns. Review the Q&A 
section to identify analyst worries and questioning patterns.

Focus on:
1. Topics repeatedly questioned by multiple analysts
2. Aggressive or skeptical questioning tone
3. Follow-up questions indicating dissatisfaction
4. New concerns not raised in previous calls
5. Questions about specific risk areas

Output format (JSON):
{
    "top_concerns": [
        {"topic": "name", "frequency": count, "analysts": ["name1", "name2"]}
    ],
    "questioning_tone": "normal/skeptical/aggressive",
    "follow_up_intensity": "high/medium/low",
    "new_concerns": ["concern1", "concern2"],
    "risk_questions": [
        {"risk_area": "name", "question": "summary", "analyst": "name"}
    ],
    "satisfaction_level": "satisfied/neutral/unsatisfied"
}

Q&A segment:
""",

    'risk_synthesizer': """
You are a senior risk analyst synthesizing insights from multiple analysis streams. 
Combine the provided analyses to identify early warning signals and prioritize risks.

Input analyses:
- Sentiment analysis results
- Topic evolution findings
- Management confidence assessment
- Analyst concern evaluation

Generate a comprehensive risk assessment with:
1. Top 3 early warning signals with severity scores
2. Risk trajectory (increasing/stable/decreasing)
3. Confidence in risk assessment
4. Recommended supervisor actions
5. Comparison to baseline risk levels

Output format (JSON):
{
    "risk_level": "green/amber/red",
    "risk_score": 0.0 to 10.0,
    "top_warnings": [
        {
            "signal": "description",
            "severity": 1-10,
            "source": "which analysis identified this",
            "evidence": "specific supporting evidence"
        }
    ],
    "risk_trajectory": "increasing/stable/decreasing",
    "confidence": 0.0 to 1.0,
    "supervisor_actions": ["action1", "action2"],
    "key_changes": "summary of changes from baseline",
    "alert_priority": "high/medium/low"
}

Analysis inputs:
"""
}

# Risk Scoring Thresholds
RISK_THRESHOLDS = {
    'green': (0, 3.5),
    'amber': (3.5, 7.0),
    'red': (7.0, 10.0)
}

# Topic Categories for Monitoring
RISK_TOPICS = [
    'credit risk',
    'market risk',
    'liquidity risk',
    'operational risk',
    'cyber risk',
    'regulatory compliance',
    'commercial real estate',
    'interest rate risk',
    'capital adequacy',
    'loan losses',
    'deposit outflows',
    'trading losses',
    'legal proceedings',
    'reputation risk'
]

# Management Language Indicators
HEDGING_WORDS = [
    'might', 'could', 'possibly', 'potentially', 'may',
    'perhaps', 'likely', 'probably', 'appears', 'seems',
    'believe', 'think', 'expect', 'anticipate', 'hope'
]

CERTAINTY_WORDS = [
    'will', 'definitely', 'certainly', 'absolutely', 'committed',
    'confident', 'ensure', 'guarantee', 'certain', 'definite'
]

# Alert Configuration
ALERT_CONFIG = {
    'sentiment_drop_threshold': -0.3,  # Alert if sentiment drops by more than this
    'confidence_drop_threshold': -0.2,  # Alert if confidence drops by more than this
    'new_risk_topics_threshold': 3,     # Alert if more than N new risk topics appear
    'analyst_concern_threshold': 0.7    # Alert if analyst concern score exceeds this
}

# Visualization Settings
CHART_COLORS = {
    'green': '#10B981',
    'amber': '#F59E0B', 
    'red': '#EF4444',
    'primary': '#3B82F6',
    'secondary': '#6B7280'
}

# Paths
DATA_PATH = os.getenv('DATA_PATH', './data/transcripts')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')
CACHE_PATH = './cache'

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'
LOG_DIR = os.getenv('LOG_DIR', 'logs')
MAX_UI_LOGS = int(os.getenv('MAX_UI_LOGS', 1000))
LOG_FILE_MAX_SIZE = int(os.getenv('LOG_FILE_MAX_SIZE', 10*1024*1024))  # 10MB
LOG_FILE_BACKUP_COUNT = int(os.getenv('LOG_FILE_BACKUP_COUNT', 5))

# Debug Console Settings
DEBUG_CONSOLE_ENABLED = os.getenv('DEBUG_CONSOLE_ENABLED', 'true').lower() == 'true'
DEBUG_CONSOLE_HEIGHT = int(os.getenv('DEBUG_CONSOLE_HEIGHT', 400))
DEBUG_CONSOLE_AUTO_SCROLL = os.getenv('DEBUG_CONSOLE_AUTO_SCROLL', 'true').lower() == 'true'

# Log Level Colors for UI
LOG_LEVEL_COLORS = {
    'DEBUG': '#6B7280',    # Gray
    'INFO': '#3B82F6',     # Blue
    'WARNING': '#F59E0B',  # Orange
    'ERROR': '#EF4444',    # Red
    'CRITICAL': '#991B1B'  # Dark Red
}