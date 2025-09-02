"""
Base Agent Class for RiskRadar Multi-Agent System
"""

import json
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from anthropic import Anthropic
import config

# Import logging functions
try:
    from src.utils.debug_logger import get_logger, log_debug, log_info, log_error, log_warning
except ImportError:
    # Fallback if logger not available
    def log_debug(msg): print(f"[DEBUG] {msg}")
    def log_info(msg): print(f"[INFO] {msg}")
    def log_error(msg): print(f"[ERROR] {msg}")
    def log_warning(msg): print(f"[WARNING] {msg}")
    def get_logger(): return None

class BaseAgent(ABC):
    """Abstract base class for all RiskRadar agents"""
    
    def __init__(self, name: str, model: str = None):
        self.name = name
        self.model = model or config.DEFAULT_MODEL
        self.prompt_template = config.AGENT_PROMPTS.get(name, "")
        self.execution_time = 0
        self.last_result = None
        self.logger = get_logger()  # Initialize logger instance
        
        log_info(f"Initializing agent: {name} with model: {self.model}")
        
        # Initialize API clients
        self.client = None
        if 'gpt' in self.model.lower():
            # Use new OpenAI client for v1.0+
            try:
                log_debug(f"Attempting to initialize OpenAI client for {self.name}")
                from openai import OpenAI
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                log_info(f"OpenAI client initialized successfully for {self.name}")
            except Exception as e:
                log_error(f"Failed to initialize OpenAI client for {self.name}: {e}")
                if self.logger:
                    self.logger.log_exception("OpenAI client initialization failed")
                self.client = None
        elif 'claude' in self.model.lower():
            try:
                log_debug(f"Attempting to initialize Anthropic client for {self.name}")
                self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                log_info(f"Anthropic client initialized successfully for {self.name}")
            except Exception as e:
                log_error(f"Failed to initialize Anthropic client for {self.name}: {e}")
                if self.logger:
                    self.logger.log_exception("Anthropic client initialization failed")
                self.client = None
        elif 'gemini' in self.model.lower():
            try:
                log_debug(f"Attempting to initialize Gemini client for {self.name}")
                log_debug(f"Model name: {self.model}")
                import google.generativeai as genai
                log_debug(f"Google GenerativeAI imported")
                genai.configure(api_key=config.GOOGLE_API_KEY)
                log_debug(f"Google API configured")
                # Gemini models don't need 'models/' prefix for initialization
                model_name = self.model.replace('models/', '')
                log_debug(f"Creating GenerativeModel with name: {model_name}")
                self.client = genai.GenerativeModel(model_name)
                log_info(f"Gemini client initialized successfully for {self.name}")
            except ImportError as ie:
                log_error(f"Import error for Gemini in {self.name}: {ie}")
                if self.logger:
                    self.logger.log_exception("Gemini import failed")
                self.client = None
            except Exception as e:
                log_error(f"Failed to initialize Gemini client for {self.name}: {e}")
                if self.logger:
                    self.logger.log_exception("Gemini client initialization failed")
                self.client = None
        
    @abstractmethod
    def analyze(self, text: str, context: Dict = None) -> Dict:
        """Analyze text and return structured results"""
        pass
    
    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM based on model configuration"""
        # Check if client is initialized
        if not self.client:
            log_warning(f"No API client available for {self.name} with model {self.model}")
            return "{}"
            
        start_time = time.time()
        
        try:
            if 'gpt' in self.model.lower():
                response = self._call_openai(prompt)
            elif 'claude' in self.model.lower():
                response = self._call_anthropic(prompt)
            elif 'gemini' in self.model.lower():
                response = self._call_gemini(prompt)
            else:
                # Default to OpenAI
                response = self._call_openai(prompt)
            
            self.execution_time = time.time() - start_time
            if self.logger:
                self.logger.log_api_call(
                    api_name=f"{self.model}",
                    model=self.model,
                    latency=self.execution_time
                )
            return response
            
        except Exception as e:
            error_msg = f"Error calling LLM for {self.name}: {e}"
            log_error(error_msg)
            if self.logger:
                self.logger.log_exception(f"LLM call failed for {self.name}")
            # Return empty dict but preserve error info for better debugging
            
            # Check for context length errors
            error_str = str(e).lower()
            if "context_length_exceeded" in error_str or "context length" in error_str or "too many tokens" in error_str or "maximum context" in error_str:
                import re
                # Try to extract token counts from error message
                token_match = re.search(r'(\d+)\s*tokens', error_str)
                limit_match = re.search(r'limit\s*(?:is\s*)?(\d+)|maximum\s*(?:is\s*)?(\d+)', error_str)
                
                token_count = token_match.group(1) if token_match else "unknown"
                limit = limit_match.group(1) or limit_match.group(2) if limit_match else "unknown"
                
                return f'{{"error": "Context length exceeded. Text has {token_count} tokens, model limit is {limit}. Consider using a model with larger context window or chunking the input."}}'
            
            # Check if it's a rate limit error
            elif "429" in str(e) or "rate" in str(e).lower():
                return '{"error": "Rate limit exceeded. Please wait a moment and try again."}'
            elif "529" in str(e) or "overloaded" in str(e).lower():
                return '{"error": "API is currently overloaded. Please try again in a few moments."}'
            else:
                return "{}"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API using v1.0+ client"""
        try:
            # Add structured output instructions
            structured_prompt = f"""{prompt}

CRITICAL: Respond with JSON wrapped in <json_response> tags:
<json_response>
{{
    "field": "value"
}}
</json_response>

No text outside the tags."""
            
            # New OpenAI API v1.0+ syntax
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analysis expert. Always respond with JSON wrapped in <json_response> tags."},
                    {"role": "user", "content": structured_prompt}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            log_error(f"OpenAI API error: {str(e)[:200]}")
            if self.logger:
                self.logger.log_exception("OpenAI API call failed")
            return "{}"
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        # Add structured output instructions
        structured_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
You MUST respond using ONLY the following format:
<json_response>
{{
    "your": "json",
    "goes": "here"
}}
</json_response>

Do NOT include ANY text outside the <json_response> tags.
Do NOT include explanations, comments, or any other text.
ONLY output the JSON between the tags."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            system="You are a financial analysis expert. You MUST respond with JSON wrapped in <json_response> tags. No other text is allowed.",
            messages=[
                {"role": "user", "content": structured_prompt}
            ]
        )
        return response.content[0].text
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        try:
            # Add structured output instructions
            full_prompt = f"""You are a financial analysis expert.

{prompt}

CRITICAL: You MUST respond with JSON wrapped in <json_response> tags:
<json_response>
{{
    "field": "value"
}}
</json_response>

No text outside the tags. Only JSON between the tags."""
            
            # Generate content
            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": config.TEMPERATURE,
                    "max_output_tokens": config.MAX_TOKENS,
                }
            )
            
            return response.text
        except Exception as e:
            log_error(f"Gemini API error: {str(e)[:200]}")
            if self.logger:
                self.logger.log_exception("Gemini API call failed")
            return "{}"
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from LLM"""
        try:
            # Clean up response if needed
            response = response.strip()
            
            # First, check for our custom XML tags
            if '<json_response>' in response and '</json_response>' in response:
                start_tag = '<json_response>'
                end_tag = '</json_response>'
                start_idx = response.index(start_tag) + len(start_tag)
                end_idx = response.index(end_tag)
                response = response[start_idx:end_idx].strip()
            
            # Handle responses that start with explanatory text
            # Look for the first '{' which indicates start of JSON
            elif '{' in response and not response.startswith('{'):
                json_start = response.index('{')
                response = response[json_start:]
            
            # Remove markdown code blocks if present
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            
            if response.endswith('```'):
                response = response[:-3]
            
            # Handle truncated responses by completing the JSON structure
            if response.count('{') > response.count('}'):
                # Count how many closing braces we need
                open_braces = response.count('{')
                close_braces = response.count('}')
                missing_braces = open_braces - close_braces
                
                # Check if we have incomplete arrays too
                open_brackets = response.count('[')
                close_brackets = response.count(']')
                missing_brackets = open_brackets - close_brackets
                
                # Add missing brackets and braces
                response = response + ']' * missing_brackets + '}' * missing_braces
            
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            log_error(f"Error parsing JSON response from {self.name}: {e}")
            log_debug(f"Response: {response[:500]}")
            
            # Try to extract JSON from the response one more time
            try:
                import re
                # Use regex to find JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            return {}
    
    def get_execution_metrics(self) -> Dict:
        """Get execution metrics for this agent"""
        return {
            'agent': self.name,
            'model': self.model,
            'execution_time': self.execution_time,
            'timestamp': time.time()
        }
    
    def validate_output(self, output: Dict) -> bool:
        """Validate agent output meets expected schema"""
        # Override in subclasses for specific validation
        return isinstance(output, dict) and len(output) > 0