"""
RiskRadar Debug Logger
Provides comprehensive logging capabilities for debugging and monitoring
"""

import logging
import sys
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import streamlit as st
from logging.handlers import RotatingFileHandler
import threading
import queue
import time
import json
import traceback
from collections import deque
import signal
import atexit

class StreamlitLogger:
    """
    Custom logger that captures all output and provides it to Streamlit UI
    Manages both file-based and session state logging
    """
    
    def __init__(self, log_dir: str = "logs", max_ui_logs: int = 1000):
        """
        Initialize the StreamlitLogger
        
        Args:
            log_dir: Directory to store log files
            max_ui_logs: Maximum number of log entries to keep in UI
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_ui_logs = max_ui_logs
        
        # Create log file path with timestamp
        self.log_filename = self.log_dir / f"riskradar_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        # Setup logging
        self._setup_logging()
        
        # Use a thread-safe deque for storing logs
        self.log_buffer = deque(maxlen=max_ui_logs)
        self.log_lock = threading.Lock()
        
        # Initialize session state for logs if not exists
        if 'debug_logs' not in st.session_state:
            st.session_state.debug_logs = []
        if 'log_filter_level' not in st.session_state:
            st.session_state.log_filter_level = 'ALL'
        if 'log_search_term' not in st.session_state:
            st.session_state.log_search_term = ''
        
        # Capture stdout and stderr
        self._setup_stream_capture()
        
    def _setup_logging(self):
        """Setup Python logging configuration"""
        # Create formatter
        self.formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        self.logger = logging.getLogger('RiskRadar')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        # Custom handler for Streamlit session state
        self.streamlit_handler = StreamlitHandler(self)
        self.streamlit_handler.setLevel(logging.DEBUG)
        self.streamlit_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.streamlit_handler)
        
    def _setup_stream_capture(self):
        """Capture stdout and stderr to also log print statements"""
        # Store original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create capture objects
        self.stdout_capture = StreamCapture(self, 'INFO', self.original_stdout)
        self.stderr_capture = StreamCapture(self, 'ERROR', self.original_stderr)
        
        # Replace streams
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        
        # Register cleanup handler for graceful shutdown
        atexit.register(self.restore_streams)
        
        # Also handle SIGINT (Ctrl+C) gracefully
        def signal_handler(signum, frame):
            self.restore_streams()
            # Re-raise to allow normal shutdown
            sys.exit(0)
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
        except:
            pass  # May fail in some environments
    
    def add_log_entry(self, log_entry: Dict[str, Any]):
        """Add log entry to buffer and session state"""
        with self.log_lock:
            self.log_buffer.append(log_entry)
            
        # Try to update session state if we're in the main thread
        try:
            # Only update if we have a valid Streamlit context
            if hasattr(st, 'session_state') and 'debug_logs' in st.session_state:
                st.session_state.debug_logs = list(self.log_buffer)
        except:
            pass  # Silently fail if not in Streamlit context
    
    def log(self, level: str, message: str, context: Optional[Dict] = None):
        """
        Log a message with given level
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            context: Optional context dictionary
        """
        # Get the appropriate logging method
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        
        # Add context to message if provided
        if context:
            message = f"{message} | Context: {json.dumps(context, default=str)}"
        
        # Log the message
        log_method(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log('DEBUG', message, kwargs if kwargs else None)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log('INFO', message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log('WARNING', message, kwargs if kwargs else None)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log('ERROR', message, kwargs if kwargs else None)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log('CRITICAL', message, kwargs if kwargs else None)
    
    def log_exception(self, message: str = "Exception occurred"):
        """Log exception with traceback"""
        self.error(f"{message}\n{traceback.format_exc()}")
    
    def log_api_call(self, api_name: str, model: str, tokens: int = 0, latency: float = 0):
        """Log API call details"""
        self.info(f"API Call: {api_name}", model=model, tokens=tokens, latency_ms=int(latency*1000))
    
    def log_agent_start(self, agent_name: str, model: str):
        """Log agent execution start"""
        self.info(f"Agent Started: {agent_name}", model=model)
    
    def log_agent_complete(self, agent_name: str, execution_time: float, success: bool = True):
        """Log agent execution completion"""
        level = 'INFO' if success else 'ERROR'
        context = {'execution_time_ms': int(execution_time*1000)}
        self.log(level, f"Agent {'Completed' if success else 'Failed'}: {agent_name}", context)
    
    def log_agent_response(self, agent_name: str, model: str, prompt: str, response: str, execution_time: float):
        """
        Log agent LLM response with truncated console output and full disk save
        
        Args:
            agent_name: Name of the agent (e.g., 'sentiment_tracker')
            model: LLM model used
            prompt: The prompt sent to the LLM
            response: The full response from the LLM
            execution_time: Time taken for the LLM call
        """
        try:
            # Import config values
            import config
            
            # Check if agent response logging is enabled
            if not getattr(config, 'DEBUG_LOG_AGENT_RESPONSES', True):
                return
            
            # Create subdirectory for agent responses
            agent_logs_subdir = getattr(config, 'DEBUG_AGENT_LOGS_SUBDIR', 'agent_responses')
            agent_logs_dir = self.log_dir / agent_logs_subdir
            agent_logs_dir.mkdir(exist_ok=True)
            
            # Generate filename with human-readable timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
            filename = f"{agent_name}_{timestamp}.json"
            filepath = agent_logs_dir / filename
            
            # Get preview lengths from config
            prompt_preview_length = getattr(config, 'DEBUG_PROMPT_PREVIEW_LENGTH', 1000)
            response_preview_length = getattr(config, 'DEBUG_RESPONSE_PREVIEW_LENGTH', 500)
            
            # Try to parse the response as JSON for pretty printing
            parsed_response = response
            try:
                # First, clean the response string, removing any surrounding text/tags
                clean_response = response.strip()
                if '<json_response>' in clean_response and '</json_response>' in clean_response:
                    start_tag = '<json_response>'
                    end_tag = '</json_response>'
                    start_idx = clean_response.find(start_tag) + len(start_tag)
                    end_idx = clean_response.find(end_tag)
                    clean_response = clean_response[start_idx:end_idx].strip()
                
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]

                # Attempt to load the cleaned JSON string into a Python dict
                parsed_response = json.loads(clean_response)
            except (json.JSONDecodeError, TypeError):
                # If it fails, it's not a valid JSON string, so keep the original raw response
                parsed_response = response

            # Prepare response data
            response_data = {
                'agent': agent_name,
                'model': model,
                'timestamp': datetime.now().isoformat(),
                'execution_time_ms': int(execution_time * 1000),
                'prompt_length': len(prompt),
                'response_length': len(response),
                'prompt': prompt.split('\n'),
                'full_response': parsed_response
            }
            
            # Save full response to disk
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            
            # Log truncated preview to console
            preview = response[:response_preview_length] + '...' if len(response) > response_preview_length else response
            
            # Remove newlines from preview for cleaner console output
            preview = preview.replace('\n', ' ').replace('\r', '')
            
            # Log summary to console
            self.info(
                f"Agent Response: {agent_name}",
                model=model,
                response_size=len(response),
                saved_to=filename
            )
            
            # Log preview at debug level
            self.debug(f"Response preview for {agent_name}: {preview}")
            
        except Exception as e:
            # Log error but don't fail the main process
            self.error(f"Failed to log agent response: {e}")
    
    def clear_logs(self):
        """Clear logs from session state"""
        with self.log_lock:
            self.log_buffer.clear()
        
        if 'debug_logs' in st.session_state:
            st.session_state.debug_logs = []
        self.info("Logs cleared by user")
    
    def get_filtered_logs(self) -> List[Dict]:
        """Get filtered logs based on current filter settings"""
        # Get logs from buffer (thread-safe copy)
        with self.log_lock:
            logs = list(self.log_buffer)
        
        # Also try to get from session state if available
        if 'debug_logs' in st.session_state and st.session_state.debug_logs:
            logs = st.session_state.debug_logs
        
        if not logs:
            return []
        
        # Filter by level
        if st.session_state.get('log_filter_level', 'ALL') != 'ALL':
            filter_level = st.session_state.log_filter_level
            logs = [log for log in logs if log.get('level') == filter_level]
        
        # Filter by search term
        search_term = st.session_state.get('log_search_term', '')
        if search_term:
            search_term = search_term.lower()
            logs = [log for log in logs if search_term in log.get('message', '').lower()]
        
        return logs
    
    def get_log_stats(self) -> Dict[str, int]:
        """Get statistics about current logs"""
        with self.log_lock:
            logs = list(self.log_buffer)
        
        # Also try to get from session state if available
        if 'debug_logs' in st.session_state and st.session_state.debug_logs:
            logs = st.session_state.debug_logs
        
        stats = {
            'DEBUG': 0,
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0,
            'CRITICAL': 0
        }
        
        for log in logs:
            level = log.get('level', 'INFO')
            if level in stats:
                stats[level] += 1
        
        return stats
    
    def export_logs(self) -> str:
        """Export logs as formatted string"""
        logs = self.get_filtered_logs()
        output = []
        for log in logs:
            output.append(f"{log.get('timestamp', '')} | {log.get('level', '')} | {log.get('message', '')}")
        return '\n'.join(output)
    
    def sync_logs_to_session(self):
        """Manually sync logs from buffer to session state"""
        with self.log_lock:
            logs = list(self.log_buffer)
        
        if 'debug_logs' in st.session_state:
            st.session_state.debug_logs = logs
    
    def restore_streams(self):
        """Restore original stdout and stderr"""
        try:
            if hasattr(self, 'original_stdout'):
                sys.stdout = self.original_stdout
            if hasattr(self, 'original_stderr'):
                sys.stderr = self.original_stderr
        except:
            # If restoration fails, just continue
            pass


class StreamlitHandler(logging.Handler):
    """Custom logging handler that sends logs to Streamlit session state"""
    
    def __init__(self, logger_instance):
        super().__init__()
        self.logger_instance = logger_instance
    
    def emit(self, record):
        """Emit a log record to session state"""
        try:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'name': record.name,
                'message': self.format(record),
                'raw_message': record.getMessage()
            }
            
            # Add to logger's buffer
            self.logger_instance.add_log_entry(log_entry)
        except:
            pass  # Silently fail to avoid recursion


class StreamCapture(io.TextIOBase):
    """Capture stdout/stderr and redirect to logger"""
    
    def __init__(self, logger_instance, level: str, original_stream):
        self.logger_instance = logger_instance
        self.level = level
        self.original_stream = original_stream
        self.buffer = []
    
    def write(self, text: str):
        """Write text to both original stream and logger"""
        try:
            # Always write to original stream first
            if self.original_stream and hasattr(self.original_stream, 'write'):
                self.original_stream.write(text)
            
            # Skip empty writes
            if not text or text == '\n':
                return
            
            # Buffer text until we have a complete line
            self.buffer.append(text)
            if '\n' in text:
                full_text = ''.join(self.buffer).strip()
                if full_text:
                    # Skip Streamlit's ScriptRunContext warnings and shutdown messages
                    if ('missing ScriptRunContext' not in full_text and 
                        'Stopping...' not in full_text and
                        'KeyboardInterrupt' not in full_text):
                        # Log the captured output
                        try:
                            self.logger_instance.log(self.level, f"[CONSOLE] {full_text}")
                        except:
                            pass  # Silently fail during shutdown
                self.buffer = []
        except:
            # During shutdown, just pass through to original stream
            if self.original_stream and hasattr(self.original_stream, 'write'):
                try:
                    self.original_stream.write(text)
                except:
                    pass
    
    def flush(self):
        """Flush the stream"""
        try:
            if self.original_stream and hasattr(self.original_stream, 'flush'):
                self.original_stream.flush()
        except:
            pass
    
    def isatty(self):
        """Check if stream is a terminal"""
        try:
            if self.original_stream and hasattr(self.original_stream, 'isatty'):
                return self.original_stream.isatty()
        except:
            pass
        return False
    
    def __getattr__(self, name):
        """Delegate any unknown attributes to the original stream"""
        # This handles cases where code expects stream attributes we haven't implemented
        if self.original_stream and hasattr(self.original_stream, name):
            return getattr(self.original_stream, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# Global logger instance
_logger_instance = None

def get_logger() -> StreamlitLogger:
    """Get or create the global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StreamlitLogger()
    return _logger_instance

def log_debug(message: str, **kwargs):
    """Convenience function for debug logging"""
    get_logger().debug(message, **kwargs)

def log_info(message: str, **kwargs):
    """Convenience function for info logging"""
    get_logger().info(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Convenience function for warning logging"""
    get_logger().warning(message, **kwargs)

def log_error(message: str, **kwargs):
    """Convenience function for error logging"""
    get_logger().error(message, **kwargs)

def log_critical(message: str, **kwargs):
    """Convenience function for critical logging"""
    get_logger().critical(message, **kwargs)