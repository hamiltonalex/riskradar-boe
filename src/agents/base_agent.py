"""
Base Agent Class for RiskRadar Multi-Agent System
"""

import json
import re
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from anthropic import Anthropic, RateLimitError
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
        self.last_prompt: Optional[str] = None
        self.last_raw_response: Optional[str] = None
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
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                log_debug(f"Google GenerativeAI imported")
                genai.configure(api_key=config.GOOGLE_API_KEY)
                log_debug(f"Google API configured")
                # Gemini models don't need 'models/' prefix for initialization
                model_name = self.model.replace('models/', '')
                log_debug(f"Creating GenerativeModel with name: {model_name}")
                self.client = genai.GenerativeModel(
                    model_name,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
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
            
            # Store prompt and response for debugging in notebooks
            self.last_prompt = prompt
            self.last_raw_response = response
            
            # Log the agent response (full response to disk, preview to console)
            if self.logger and response:
                self.logger.log_agent_response(
                    agent_name=self.name,
                    model=self.model,
                    prompt=prompt,
                    response=response,
                    execution_time=self.execution_time
                )
            
            # Also log the API call metrics
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
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a text string.
        Rough estimate: 1 token ≈ 4 characters for English text.
        """
        return len(text) // 4

    def _calculate_safe_max_tokens(self, prompt: str, model_context_limit: int = 128000) -> int:
        """
        Calculate safe max_tokens based on prompt length.
        Leaves buffer for model processing.

        Args:
            prompt: The input prompt text
            model_context_limit: The model's total context window size

        Returns:
            Safe number of max tokens for completion
        """
        # Estimate input tokens
        estimated_input_tokens = self._estimate_token_count(prompt)

        # For GPT-5 models, use a different calculation
        # GPT-5 has separate input/output limits, not a combined context
        if 'gpt-5' in self.model.lower():
            # GPT-5 can handle up to 128K output tokens
            # But we want to be reasonable - use config.MAX_TOKENS as target
            # Don't artificially cap it too low
            safe_max = min(
                config.MAX_TOKENS,
                128000  # GPT-5 output limit
            )
        else:
            # For other models, use combined context window calculation
            # Leave 20% buffer and ensure reasonable minimum for output
            safe_max = min(
                config.MAX_TOKENS,
                max(1000, model_context_limit - estimated_input_tokens - int(model_context_limit * 0.2))
            )

        # Log for debugging
        from src.utils.debug_logger import log_debug
        log_debug(f"{self.name}: Input ~{estimated_input_tokens} tokens, setting max_completion to {safe_max}")

        return safe_max

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API using v1.0+ client with retry logic"""
        # Add structured output instructions
        structured_prompt = f"""{prompt}

CRITICAL: Respond with JSON wrapped in <json_response> tags:
<json_response>
{{
    "field": "value"
}}
</json_response>

No text outside the tags."""

        # Define fallback token limits for retries
        token_limits = [config.MAX_TOKENS, 4000, 2000, 1000]

        for attempt, max_tokens in enumerate(token_limits):
            try:
                # OpenAI API v1.0+ syntax
                # GPT-5 models have special requirements
                if 'gpt-5' in self.model.lower():
                    # For GPT-5, use the current retry limit
                    safe_max_tokens = max_tokens

                    if attempt > 0:
                        log_info(f"{self.name}: Retry {attempt} with max_tokens={safe_max_tokens}")

                    # GPT-5 requires max_completion_tokens and only supports default temperature (1.0)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a financial analysis expert. Always respond with JSON wrapped in <json_response> tags."},
                            {"role": "user", "content": structured_prompt}
                        ],
                        max_completion_tokens=safe_max_tokens
                        # temperature parameter omitted - GPT-5 only supports default (1.0)
                    )
                else:
                    # Older models use max_tokens and support custom temperature
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a financial analysis expert. Always respond with JSON wrapped in <json_response> tags."},
                            {"role": "user", "content": structured_prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=config.TEMPERATURE
                    )

                # Check for content refusal or empty response
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = choice.finish_reason

                # Check finish_reason first for better error handling
                if finish_reason == "length":
                    # Response was truncated due to length limit
                    log_warning(f"{self.name}: Response truncated due to length limit with max_tokens={max_tokens}")

                    # If we got partial content, try to use it
                    if content and content.strip():
                        log_info(f"{self.name}: Attempting to parse partial response...")
                        # Return the partial content and let parsing handle it
                        return content
                    else:
                        # No content at all - retry with smaller limit if available
                        if attempt < len(token_limits) - 1:
                            log_warning(f"{self.name}: No content received, will retry with smaller token limit")
                            continue  # Try next token limit
                        else:
                            # Final attempt failed
                            input_tokens = self._estimate_token_count(structured_prompt)
                            log_error(f"{self.name}: All retry attempts failed. Input: ~{input_tokens} tokens")
                            return json.dumps({
                                "error": "Response exceeded token limit after all retries.",
                                "status": "truncated",
                                "input_tokens_estimate": input_tokens,
                                "attempts": len(token_limits),
                                "suggestion": "Document may be too large or model may have issues"
                            })

                # Check if content was refused or filtered
                if content is None or (isinstance(content, str) and content.strip() == ""):
                    refusal = getattr(choice.message, 'refusal', None)

                    if refusal:
                        # GPT-5 explicit refusal
                        log_warning(f"{self.name}: Model refused request: {refusal}")
                        return json.dumps({
                            "error": f"Model refused request: {refusal}",
                            "status": "refused"
                        })
                    elif finish_reason == "content_filter":
                        # Content filtered
                        log_warning(f"{self.name}: Response blocked by content filter")
                        return json.dumps({
                            "error": "Response blocked by content filter. Try a different model or rephrase.",
                            "status": "filtered"
                        })
                    else:
                        # Empty for unknown reason - retry if possible
                        if attempt < len(token_limits) - 1:
                            log_warning(f"{self.name}: Empty response (finish_reason: {finish_reason}), retrying...")
                            continue
                        else:
                            log_warning(f"{self.name}: Empty response after all retries (finish_reason: {finish_reason})")
                            return json.dumps({
                                "error": f"Model returned empty response (finish_reason: {finish_reason})",
                                "status": "empty",
                                "attempts": len(token_limits)
                            })

                # Success - return the content
                return content

            except Exception as e:
                # Log error and try next token limit if available
                log_error(f"OpenAI API error with max_tokens={max_tokens}: {str(e)[:200]}")
                if attempt < len(token_limits) - 1:
                    log_info(f"{self.name}: Will retry with smaller token limit")
                    continue
                else:
                    # Final error after all retries
                    if self.logger:
                        self.logger.log_exception("OpenAI API call failed after all retries")
                    return json.dumps({
                        "error": f"API call failed: {str(e)[:100]}",
                        "status": "api_error",
                        "attempts": len(token_limits)
                    })

        # Should never reach here, but just in case
        return "{}"
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API with rate limit handling"""
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
        
        max_retries = 3
        base_delay = 10
        
        for attempt in range(max_retries):
            try:
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
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    if self.logger:
                        from src.utils.debug_logger import log_warning
                        log_warning(f"Rate limit reached for {self.name}. Waiting {delay} seconds before retry (attempt {attempt + 1}/{max_retries})")

                    # Streamlit UI updates removed to prevent ScriptRunContext warnings
                    # when running in ThreadPoolExecutor worker threads.
                    # Rate limit info is logged above for debugging.

                    time.sleep(delay)
                else:
                    if self.logger:
                        from src.utils.debug_logger import log_error
                        log_error(f"Rate limit exceeded for {self.name} after {max_retries} attempts")
                    return '{"error": "Rate limit exceeded. Please try again later.", "status": "rate_limited"}'
    
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

            # Check candidate for finish reason
            finish_reason_val = response.candidates[0].finish_reason if response.candidates else None
            
            # Check if response was blocked or has no content
            if not response.parts:
                reason = "Unknown"
                if finish_reason_val:
                    reason = finish_reason_val.name
                # Check prompt feedback as a fallback
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name

                log_warning(f"Gemini response for {self.name} was empty. Finish reason: {reason}")
                return f'{{"error": "The response from the model was blocked or empty.", "finish_reason": "{reason}"}}'

            # Check if the response was truncated
            if finish_reason_val and finish_reason_val.name == 'MAX_TOKENS':
                log_warning(f"Gemini response for {self.name} was truncated. The JSON may be incomplete.")
                # We will still return the text and let the robust parser try to handle it.
            
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
            
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            log_error(f"Error parsing JSON response from {self.name}: {e}")
            log_debug(f"Response: {response[:500]}")

            try:
                log_debug("Attempting robust JSON repair...")
                # Find the last comma or opening brace/bracket
                last_comma = response.rfind(',')
                last_open_brace = response.rfind('{')
                last_open_bracket = response.rfind('[')
                
                # Determine the point to truncate to
                cut_off_point = max(last_comma, last_open_brace, last_open_bracket)
                
                if cut_off_point > 0:
                    # Trim the string to remove the partial element
                    repaired_response = response[:cut_off_point]
                    
                    # Balance braces and brackets
                    open_braces = repaired_response.count('{')
                    close_braces = repaired_response.count('}')
                    open_brackets = repaired_response.count('[')
                    close_brackets = repaired_response.count(']')
                    
                    repaired_response += ']' * (open_brackets - close_brackets)
                    repaired_response += '}' * (open_braces - close_braces)
                    
                    # Final check for valid structure before parsing
                    if repaired_response.startswith('{') and repaired_response.endswith('}'):
                        log_debug(f"Repaired JSON attempt: {repaired_response[:500]}")
                        return json.loads(repaired_response)
            except Exception as repair_e:
                log_error(f"Robust JSON repair failed: {repair_e}")
                # Fall through to the final regex attempt
            
            # Try to extract JSON from the response one more time with regex as a last resort
            try:
                # Use regex to find the largest valid-looking JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    # Try to parse the matched string
                    return json.loads(json_match.group())
            except Exception as final_e:
                log_error(f"Final attempt to parse JSON with regex failed: {final_e}")
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