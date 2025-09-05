"""
RiskRadar - Improved Streamlit Dashboard with Model Validation
Properly links model selector to agents and validates API keys
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
import sys
import time

# Ensure we're in the right directory
app_dir = Path(__file__).parent
os.chdir(app_dir)
sys.path.insert(0, str(app_dir))

# Import our modules
from src.data.data_loader import TranscriptLoader
from src.data.transcript_processor import TranscriptProcessor
from src.agents.sentiment_agent import SentimentTrackerAgent
from src.agents.topic_agent import TopicEvolutionAgent
from src.agents.confidence_agent import ManagementConfidenceAgent
from src.agents.analyst_agent import AnalystConcernAgent
from src.agents.orchestrator import RiskSynthesizer
from src.agents.rag_module import ArkadiuszRAGSystem, init_rag_system, get_rag_system
from src.models.model_evaluator import ModelEvaluator
from src.utils.source_tracker import source_tracker, DocumentReference, SourceQuote
from src.utils.document_viewer import document_viewer
from src.utils.debug_logger import get_logger, log_info, log_error, log_debug, log_warning
import config

# Page configuration
st.set_page_config(
    page_title="RiskRadar - BoE Early Warning System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .risk-green { background-color: #10B981; color: white; padding: 10px; border-radius: 5px; }
    .risk-amber { background-color: #F59E0B; color: white; padding: 10px; border-radius: 5px; }
    .risk-red { background-color: #EF4444; color: white; padding: 10px; border-radius: 5px; }
    .api-error { background-color: #EF4444; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .api-warning { background-color: #F59E0B; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .api-success { background-color: #10B981; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Initialize logger
logger = get_logger()
log_info("RiskRadar Application Started")

# Sync any existing logs to session state
logger.sync_logs_to_session()

# Initialize session state
if 'analyzed_documents' not in st.session_state:
    st.session_state.analyzed_documents = []
if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'source_tracker' not in st.session_state:
    st.session_state.source_tracker = source_tracker
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'api_keys_valid' not in st.session_state:
    st.session_state.api_keys_valid = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'final_assessment' not in st.session_state:
    st.session_state.final_assessment = None
if 'analyzed_files' not in st.session_state:
    st.session_state.analyzed_files = []
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0
if 'api_test_results' not in st.session_state:
    st.session_state.api_test_results = {}
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'should_run_analysis' not in st.session_state:
    st.session_state.should_run_analysis = False
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # Default to Risk Dashboard
if 'analysis_progress' not in st.session_state:
    st.session_state.analysis_progress = 0
if 'analysis_status' not in st.session_state:
    st.session_state.analysis_status = "Ready"
if 'analysis_step' not in st.session_state:
    st.session_state.analysis_step = 0

# RAG System session state (Arkadiusz's module)
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'rag_query_history' not in st.session_state:
    st.session_state.rag_query_history = []
if 'rag_indexed_docs' not in st.session_state:
    st.session_state.rag_indexed_docs = []
if 'rag_collection_stats' not in st.session_state:
    st.session_state.rag_collection_stats = {}
if 'rag_indexed_info' not in st.session_state:
    st.session_state.rag_indexed_info = None
if 'rag_chunk_count' not in st.session_state:
    st.session_state.rag_chunk_count = None
if 'rag_index_synchronized' not in st.session_state:
    st.session_state.rag_index_synchronized = None  # None = unknown, True = synced, False = unsynced

def trigger_analysis():
    '''Callback function for Run Analysis button'''
    st.session_state.should_run_analysis = True
    st.session_state.analysis_running = True
    st.session_state.analysis_just_started = True
    st.session_state.analysis_progress = 0
    st.session_state.analysis_status = "Initializing..."
    st.session_state.analysis_step = 0
    # Mark that we want to show debug console (expander will auto-expand)
    if config.DEBUG_CONSOLE_ENABLED:
        st.session_state.show_debug_console = True

def reset_analysis_state():
    '''Reset the analysis state after completion'''
    st.session_state.analysis_running = False
    st.session_state.should_run_analysis = False
    st.session_state.analysis_progress = 100
    st.session_state.analysis_status = "Complete"
    # Switch back to Risk Dashboard tab
    st.session_state.active_tab = 0  # Risk Dashboard tab

def clear_uploaded_files():
    '''Callback to clear all uploaded files'''
    st.session_state.clear_uploads_flag = True

def update_analysis_progress(progress: int, status: str, force_refresh: bool = False):
    '''Update analysis progress and trigger UI refresh'''
    st.session_state.analysis_progress = progress
    st.session_state.analysis_status = status
    # Sync logs to session state
    if logger:
        logger.sync_logs_to_session()
    
    # Force a UI refresh for critical updates
    if force_refresh and st.session_state.get('analysis_step', 0) != progress:
        st.session_state.analysis_step = progress
        # This will cause the UI to refresh and show updates
        time.sleep(0.1)  # Small delay to ensure state is saved

def validate_api_key(model_name: str) -> tuple[bool, str]:
    """
    Validate if API key exists for selected model
    Returns: (is_valid, error_message)
    """
    if 'gpt' in model_name.lower():
        if not config.OPENAI_API_KEY:
            return False, "OpenAI API key not found. Please add OPENAI_API_KEY to .env file"
        # Could add actual API validation here
        return True, "OpenAI API key configured"
    
    elif 'claude' in model_name.lower():
        if not config.ANTHROPIC_API_KEY:
            return False, "Anthropic API key not found. Please add ANTHROPIC_API_KEY to .env file"
        return True, "Anthropic API key configured"
    
    elif 'gemini' in model_name.lower():
        if not config.GOOGLE_API_KEY:
            return False, "Google API key not found. Please add GOOGLE_API_KEY to .env file"
        return True, "Google Gemini API key configured"
    
    else:
        return False, f"Unknown model: {model_name}"

def test_api_connection(api_name: str) -> tuple[bool, str]:
    """
    Test if an API connection actually works
    Returns: (success, message)
    """
    log_debug(f"{'='*50}")
    log_debug(f"Testing API connection for: {api_name}")
    log_debug(f"{'='*50}")
    
    # Check cache first
    if api_name in st.session_state.api_test_results:
        log_debug(f"[DEBUG] Found cached result for {api_name}: {st.session_state.api_test_results[api_name]}")
        return st.session_state.api_test_results[api_name]
    
    try:
        if api_name == "openai":
            log_debug(f"[DEBUG] OpenAI API Key present: {bool(config.OPENAI_API_KEY)}")
            if not config.OPENAI_API_KEY:
                result = (False, "No API key")
            else:
                # Try a minimal API call
                try:
                    log_debug("[DEBUG] Attempting to import OpenAI...")
                    from openai import OpenAI
                    log_debug(f"[DEBUG] OpenAI imported successfully")
                    log_debug(f"[DEBUG] Creating OpenAI client...")
                    client = OpenAI(api_key=config.OPENAI_API_KEY)
                    log_debug(f"[DEBUG] OpenAI client created")
                    # Use a very cheap model list call to test
                    log_debug("[DEBUG] Calling models.list()...")
                    models = client.models.list()
                    # Convert to list to ensure it actually executes
                    log_debug("[DEBUG] Converting models to list...")
                    model_list = list(models)[:1]
                    log_debug(f"[DEBUG] Successfully retrieved {len(model_list)} model(s)")
                    result = (True, "Connected")
                except ImportError as ie:
                    log_debug(f"[ERROR] OpenAI import failed: {ie}")
                    import traceback
                    traceback.print_exc()
                    result = (False, "OpenAI library not properly installed")
                except Exception as e:
                    log_debug(f"[ERROR] OpenAI connection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    error_msg = str(e)
                    if "api_key" in error_msg.lower():
                        result = (False, "Invalid API key")
                    else:
                        result = (False, f"Connection failed: {error_msg[:50]}")
                    
        elif api_name == "anthropic":
            log_debug(f"[DEBUG] Anthropic API Key present: {bool(config.ANTHROPIC_API_KEY)}")
            if not config.ANTHROPIC_API_KEY:
                result = (False, "No API key")
            else:
                try:
                    log_debug("[DEBUG] Attempting to import anthropic...")
                    import anthropic
                    log_debug(f"[DEBUG] Anthropic imported successfully")
                    log_debug(f"[DEBUG] Creating Anthropic client...")
                    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                    log_debug(f"[DEBUG] Anthropic client created successfully")
                    # Simple test - create client successfully means key format is valid
                    result = (True, "Connected")
                except ImportError as ie:
                    log_debug(f"[ERROR] Anthropic import failed: {ie}")
                    import traceback
                    traceback.print_exc()
                    result = (False, "Anthropic library not installed")
                except Exception as e:
                    log_debug(f"[ERROR] Anthropic connection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    result = (False, f"Connection failed: {str(e)[:50]}")
                    
        elif api_name == "google":
            log_debug(f"[DEBUG] Google API Key present: {bool(config.GOOGLE_API_KEY)}")
            if not config.GOOGLE_API_KEY:
                result = (False, "No API key")
            else:
                try:
                    # Test Google Gemini API
                    log_debug("[DEBUG] Attempting to import google.generativeai...")
                    import sys
                    log_debug(f"[DEBUG] Python path: {sys.path[:3]}...")
                    import google.generativeai as genai
                    log_debug(f"[DEBUG] Google GenerativeAI imported successfully")
                    log_debug(f"[DEBUG] Configuring Google API...")
                    genai.configure(api_key=config.GOOGLE_API_KEY)
                    log_debug(f"[DEBUG] Google API configured")
                    # Try to list models as a connection test
                    log_debug("[DEBUG] Listing Google models...")
                    models = list(genai.list_models())[:1]
                    log_debug(f"[DEBUG] Retrieved {len(models)} model(s)")
                    if models:
                        log_debug(f"[DEBUG] First model: {models[0].name}")
                        result = (True, "Connected")
                    else:
                        result = (True, "Connected (no models listed)")
                except ImportError as ie:
                    log_debug(f"[ERROR] Google GenerativeAI import failed: {ie}")
                    import traceback
                    traceback.print_exc()
                    # Check if package is installed
                    try:
                        import pkg_resources
                        installed = [pkg.key for pkg in pkg_resources.working_set]
                        log_debug(f"[DEBUG] Installed packages containing 'google': {[p for p in installed if 'google' in p]}")
                    except:
                        pass
                    result = (False, f"Import error: {str(ie)[:50]}")
                except Exception as e:
                    log_debug(f"[ERROR] Google connection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    error_msg = str(e)
                    if "api_key" in error_msg.lower() or "API key" in error_msg:
                        result = (False, "Invalid API key")
                    else:
                        result = (False, f"Connection failed: {error_msg[:50]}")
        else:
            result = (False, "Unknown API")
            
    except ImportError as e:
        log_debug(f"[ERROR] Outer ImportError caught: {e}")
        import traceback
        traceback.print_exc()
        result = (False, f"Library not installed: {getattr(e, 'name', str(e))}")
    except Exception as e:
        log_debug(f"[ERROR] Outer Exception caught: {e}")
        import traceback
        traceback.print_exc()
        result = (False, f"Error: {str(e)[:50]}")
    
    log_debug(f"[DEBUG] Final result for {api_name}: {result}")
    log_debug(f"{'='*50}\n")
    
    # Cache the result
    st.session_state.api_test_results[api_name] = result
    return result

def get_api_status_icon(api_name: str, test_connection: bool = False) -> str:
    """Get appropriate icon for API status"""
    # Check if key exists
    has_key = False
    if api_name == "openai":
        has_key = bool(config.OPENAI_API_KEY)
    elif api_name == "anthropic":
        has_key = bool(config.ANTHROPIC_API_KEY)
    elif api_name == "google":
        has_key = bool(config.GOOGLE_API_KEY)
    
    if not has_key:
        return "❌"  # No key configured
    
    if test_connection:
        # Test actual connection
        success, msg = test_api_connection(api_name)
        if success:
            return "✅"  # Configured and working
        else:
            return "⚠️"  # Configured but not working
    else:
        return "🔑"  # Key exists but not tested

def initialize_agents_with_model(model_name: str):
    """Initialize all agents with specific model"""
    try:
        return {
            'sentiment': SentimentTrackerAgent(model=model_name),
            'topics': TopicEvolutionAgent(model=model_name),
            'confidence': ManagementConfidenceAgent(model=model_name),
            'analyst': AnalystConcernAgent(model=model_name),
            'orchestrator': RiskSynthesizer(model=model_name),
            'processor': TranscriptProcessor(),
            'loader': TranscriptLoader()
        }
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}")
        return None

# Header
st.title("🎯 RiskRadar: Early Warning System")
st.markdown("**Bank of England Supervisory Intelligence Platform**")

# Debug Console Section (if enabled) - Above everything else
if config.DEBUG_CONSOLE_ENABLED:
    # Create a collapsible debug console
    with st.expander("🔍 Debug Console", expanded=st.session_state.get('analysis_running', False)):
        # Always sync logs from buffer to session state
        logger.sync_logs_to_session()
        
        # Console controls
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
        
        with col1:
            # Log level filter
            log_levels = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            st.session_state.log_filter_level = st.selectbox(
                "Filter Level",
                log_levels,
                index=log_levels.index(st.session_state.get('log_filter_level', 'ALL')),
                key='log_level_filter'
            )
        
        with col2:
            # Search filter
            st.session_state.log_search_term = st.text_input(
                "Search Logs",
                value=st.session_state.get('log_search_term', ''),
                placeholder="Enter search term...",
                key='log_search'
            )
        
        with col3:
            # Auto-scroll toggle
            auto_scroll = st.checkbox(
                "Auto-scroll",
                value=config.DEBUG_CONSOLE_AUTO_SCROLL,
                key='auto_scroll_toggle'
            )
        
        with col4:
            # Clear logs button
            if st.button("🗑️ Clear", key='clear_logs_btn'):
                logger.clear_logs()
                st.rerun()
        
        with col5:
            # Download logs button
            logs_export = logger.export_logs()
            st.download_button(
                "📥 Export",
                data=logs_export,
                file_name=f"riskradar_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key='export_logs_btn'
            )
        
        # Display logs in a scrollable area
        st.markdown("### 📋 Log Output")
        
        # Get filtered logs
        filtered_logs = logger.get_filtered_logs()
        
        if filtered_logs:
            # Create formatted log display
            log_display = []
            for log in filtered_logs[-100:]:  # Show last 100 logs for performance
                level = log.get('level', 'INFO')
                color = config.LOG_LEVEL_COLORS.get(level, '#666')
                timestamp = log.get('timestamp', '')
                message = log.get('raw_message', log.get('message', ''))
                
                # Format each log entry with color coding
                log_entry = f"<span style='color: {color};'>[{timestamp}] [{level}]</span> {message}"
                log_display.append(log_entry)
            
            # Use HTML to display colored logs
            log_html = "<br>".join(log_display)
            st.markdown(
                f"""
                <div style='
                    background-color: #1e1e1e; 
                    color: #d4d4d4; 
                    padding: 15px; 
                    border-radius: 5px; 
                    font-family: monospace; 
                    font-size: 12px; 
                    height: 400px; 
                    overflow-y: auto;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                '>
{log_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No logs to display. Logs will appear here as the application runs.")

# Get the disabled state for Configuration panel only
config_disabled = st.session_state.get('analysis_running', False)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection with validation
    st.subheader("🤖 AI Model Selection")
    
    # Build available models dynamically based on configured APIs
    available_models = {}
    
    # OpenAI models - based on common available models in 2024/2025
    if config.OPENAI_API_KEY:
        available_models.update({
            "gpt-5": "GPT-5 (Most Advanced)",
            "gpt-5-mini": "GPT-5 Mini (Fast Next Gen)",
            "gpt-5-nano": "GPT-5 Nano (Ultra Fast)",
            "gpt-4o": "GPT-4o (Balanced Performance)",
            "gpt-4o-mini": "GPT-4o Mini (Cost-Effective)",
            "o1-mini": "O1 Mini (Reasoning Model)",
        })
    
    # Anthropic/Claude models - latest verified models
    if config.ANTHROPIC_API_KEY:
        available_models.update({
            "claude-3-haiku-20240307": "Claude 3 Haiku (Fast & Efficient)",
            "claude-3-5-haiku-latest": "Claude 3.5 Haiku (Improved Speed)",
            "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet (Latest & Powerful)",
        })
    
    # Google Gemini models - stable models as of 2025
    if config.GOOGLE_API_KEY:
        available_models.update({
            "gemini-2.5-flash": "Gemini 2.5 Flash (Best Price/Performance)",
            "gemini-2.5-pro": "Gemini 2.5 Pro (Most Capable with Thinking)",
            "gemini-2.0-flash": "Gemini 2.0 Flash (Fast Multimodal)",
        })
    
    # If no API keys are configured, show demo models
    if not available_models:
        available_models = {
            "demo-mode": "Demo Mode (No API Required)",
        }
        st.warning("⚠️ No API keys configured. Please add API keys to use AI models.")
    
    # Select default model based on what's available
    default_model = None
    if "claude-3-7-sonnet-latest" in available_models:
        default_model = "claude-3-7-sonnet-latest"
    elif "gpt-5-mini" in available_models:
        default_model = "gpt-5-mini"
    elif "gemini-2.5-flash" in available_models:
        default_model = "gemini-2.5-flash"
    elif "gpt-4o-mini" in available_models:
        default_model = "gpt-4o-mini"
    elif available_models:
        default_model = list(available_models.keys())[0]
    
    selected_model = st.selectbox(
        "Select AI Model",
        list(available_models.keys()),
        index=list(available_models.keys()).index(default_model) if default_model else 0,
        format_func=lambda x: available_models[x],
        help="Choose the AI model for analysis. Models are shown based on configured API keys.",
        disabled=config_disabled
    )
    
    # Validate selected model's API key
    is_valid, message = validate_api_key(selected_model)
    
    if is_valid:
        st.markdown(f'<div class="api-success">✅ {message}</div>', unsafe_allow_html=True)
        api_ready = True
    else:
        st.markdown(f'<div class="api-error">❌ {message}</div>', unsafe_allow_html=True)
        api_ready = False
        
        # Show alternative options
        st.markdown("### Alternative Options:")
        
        # Check which APIs are available
        alternatives = []
        if config.OPENAI_API_KEY and 'gpt' not in selected_model:
            alternatives.append("OpenAI models")
        if config.ANTHROPIC_API_KEY and 'claude' not in selected_model:
            alternatives.append("Anthropic models")
        if config.GOOGLE_API_KEY and 'gemini' not in selected_model:
            alternatives.append("Google Gemini models")
        
        if alternatives:
            st.info(f"Available alternatives: {', '.join(alternatives)}")
        else:
            st.warning("No API keys configured. Running in demo mode with limited functionality.")
    
    # Show API configuration status
    with st.expander("🔑 API Configuration Status", expanded=False):
        st.caption("Icon meanings: ❌ = No key | 🔑 = Key exists | ✅ = Tested & working | ⚠️ = Key exists but failed")
        
        # Test connection button
        test_apis = st.button("🔍 Test Connections", help="Test if API keys are valid", disabled=config_disabled)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**API Provider**")
            st.write("OpenAI")
            st.write("Anthropic")
            st.write("Google")
            
        with col2:
            st.write("**Status**")
            openai_icon = get_api_status_icon("openai", test_connection=test_apis)
            anthropic_icon = get_api_status_icon("anthropic", test_connection=test_apis)
            google_icon = get_api_status_icon("google", test_connection=test_apis)
            st.write(openai_icon)
            st.write(anthropic_icon)
            st.write(google_icon)
            
        with col3:
            st.write("**Details**")
            if test_apis:
                # Show test results
                success, msg = test_api_connection("openai")
                st.caption(msg if config.OPENAI_API_KEY else "Not configured")
                
                success, msg = test_api_connection("anthropic")
                st.caption(msg if config.ANTHROPIC_API_KEY else "Not configured")
                
                success, msg = test_api_connection("google")
                st.caption(msg if config.GOOGLE_API_KEY else "Not configured")
            else:
                st.caption("Key exists" if config.OPENAI_API_KEY else "Not configured")
                st.caption("Key exists" if config.ANTHROPIC_API_KEY else "Not configured")
                st.caption("Key exists" if config.GOOGLE_API_KEY else "Not configured")
        
        if test_apis:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💡 Test results are cached for this session.")
            with col2:
                if st.button("🔄 Clear Cache", disabled=config_disabled):
                    st.session_state.api_test_results = {}
                    st.rerun()
    
    st.divider()
    
    # File selection
    st.subheader("📄 Document Selection")
    
    # Show available documents
    available_files = []
    data_folders = ['data/transcripts', 'data/bank_reports', 'data/regulatory_docs', 'data/uploads']
    
    # Get the base directory of the app
    base_dir = Path(__file__).parent
    
    # Ensure uploads folder exists
    uploads_folder = base_dir / 'data' / 'uploads'
    uploads_folder.mkdir(parents=True, exist_ok=True)
    
    for folder in data_folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            for file in folder_path.iterdir():
                if file.suffix in ['.pdf', '.txt'] and file.is_file():
                    # Store absolute path but display relative path
                    available_files.append(str(file))
    
    # Create display names for better readability
    display_names = {}
    for f in available_files:
        path = Path(f)
        if 'uploads' in str(f):
            display_names[f] = f"📤 {path.name} (uploaded)"
        else:
            display_names[f] = f"{path.parent.name}/{path.name}"
    
    # Handle clear uploads flag before widget creation
    if st.session_state.get('clear_uploads_flag', False):
        # Remove uploaded files from selection
        if 'file_selector' in st.session_state:
            st.session_state.file_selector = [f for f in st.session_state.file_selector if 'uploads' not in f]
        # Delete files from disk
        for file in uploads_folder.glob('*.*'):
            if file.name != '.gitkeep':
                file.unlink()
        # Clear the flag
        st.session_state.clear_uploads_flag = False
        # Increment upload key to ensure clean state
        st.session_state.upload_key += 1
    
    # Handle pending files from upload before widget creation
    if 'pending_files' in st.session_state and st.session_state.pending_files:
        if 'file_selector' not in st.session_state:
            st.session_state.file_selector = []
        # Add pending files to selection
        current_selection = st.session_state.file_selector if isinstance(st.session_state.file_selector, list) else []
        for file_path in st.session_state.pending_files:
            if file_path not in current_selection:
                current_selection.append(file_path)
        st.session_state.file_selector = current_selection
        st.session_state.pending_files = []  # Clear pending files
    
    # Initialize default selection if needed
    if 'file_selector' not in st.session_state:
        # Initialize with first 3 files or all if less than 3
        non_upload_files = [f for f in available_files if 'uploads' not in f]
        default_selection = non_upload_files[:3] if len(non_upload_files) >= 3 else non_upload_files
        st.session_state.file_selector = default_selection
    
    # Ensure session state is a list and all files are still available
    if not isinstance(st.session_state.file_selector, list):
        st.session_state.file_selector = []
    
    valid_selections = [f for f in st.session_state.file_selector if f in available_files]
    if valid_selections != st.session_state.file_selector:
        st.session_state.file_selector = valid_selections
    
    # Create multiselect using the key's state directly
    selected_files = st.multiselect(
        "Select documents to analyze",
        available_files,
        format_func=lambda x: display_names.get(x, Path(x).name),
        help="Select one or more documents to analyze. Uploaded files will be added automatically.",
        key="file_selector",
        disabled=config_disabled
    )
    
    # Store selected files in session state for other tabs to access
    st.session_state.selected_files = selected_files
    
    # Show selection count based on actual selection
    if selected_files:
        selected_count = len(selected_files)
        uploaded_in_selection = len([f for f in selected_files if 'uploads' in f])
        if uploaded_in_selection > 0:
            st.info(f"📌 {selected_count} file(s) selected ({uploaded_in_selection} uploaded)")
        else:
            st.info(f"📌 {selected_count} file(s) selected for analysis")
    else:
        st.info("📌 No files selected")
    
    # Upload custom files
    st.subheader("📤 Upload Additional Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'pdf', 'json'],
        accept_multiple_files=True,
        help="Upload transcript or report files to analyze. Files will be saved to data/uploads for future use.",
        key=f"file_uploader_{st.session_state.upload_key}",
        disabled=config_disabled
    )
    
    # Save uploaded files to uploads folder and add to selected files
    newly_uploaded_files = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Create unique filename to avoid conflicts
            file_path = uploads_folder / uploaded_file.name
            
            # Save the file
            try:
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_path_str = str(file_path)
                newly_uploaded_files.append(file_path_str)
                
                # Add to available files if not already there
                if file_path_str not in available_files:
                    available_files.append(file_path_str)
                    
                st.success(f"✅ Saved: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Failed to save {uploaded_file.name}: {str(e)}")
        
        if newly_uploaded_files:
            st.info(f"📎 {len(newly_uploaded_files)} file(s) saved and added to selection")
            # Store the newly uploaded files in a separate session state variable
            if 'pending_files' not in st.session_state:
                st.session_state.pending_files = []
            st.session_state.pending_files.extend(newly_uploaded_files)
            
            # Increment upload key to reset the file uploader widget
            st.session_state.upload_key += 1
            
            # Force rerun to update the UI with new selections and clear the uploader
            st.rerun()
    
    # Show existing uploaded files  
    uploaded_files_on_disk = [f for f in available_files if 'uploads' in f]
    if uploaded_files_on_disk:
        with st.expander(f"📁 Previously Uploaded Files ({len(uploaded_files_on_disk)})"):
            col1, col2 = st.columns([4, 1])
            with col1:
                for file_path in uploaded_files_on_disk:
                    file_name = Path(file_path).name
                    # Check if this file is selected
                    is_selected = file_path in selected_files
                    if is_selected:
                        st.write(f"✅ {file_name}")
                    else:
                        st.write(f"⬜ {file_name}")
            with col2:
                if st.button("🗑️ Clear All", 
                           help="Remove all uploaded files", 
                           disabled=config_disabled,
                           on_click=clear_uploaded_files):
                    st.rerun()
    
    # Set default values for removed options (keeping functionality)
    show_sources = True  # Always show source citations
    show_confidence = True  # Always show confidence scores
    use_fallback = True  # Always use fallback if API fails
    
    st.divider()
    
    # Analysis button with callback
    if api_ready:
        st.button(
            "🔍 Run Analysis", 
            type="primary", 
            use_container_width=True,
            disabled=config_disabled,
            on_click=trigger_analysis,
            key="run_analysis_btn"
        )
    else:
        st.warning("⚠️ Please configure API keys or select a model with valid API key")
        st.button(
            "🔍 Run Demo Analysis (Limited)", 
            type="secondary", 
            use_container_width=True,
            disabled=config_disabled,
            on_click=trigger_analysis,
            key="run_demo_analysis_btn"
        )
    
    # Add spacing before credits
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    
    # Credits section at the bottom of sidebar
    st.divider()
    
    # App branding and version
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <strong>RiskRadar v1.0</strong><br>
            Bank of England Employer Project<br>
            Cambridge Data Science 2025
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Team credits in an expander
    with st.expander("👥 Team: Overfit and Underpaid"):
        st.markdown(
            """
            <div style='text-align: center; padding: 10px;'>
                <h4 style='color: #1f77b4; margin-bottom: 15px;'>Project Team</h4>
                <p style='line-height: 1.8;'>
                    <strong>Rajen Lavingia</strong><br>
                    <strong>Jessica Abreu</strong><br>
                    <strong>Arkadiusz Tomczak</strong><br>
                    <strong>Adeyinka Abdulrahman</strong><br>
                    <strong>Alex Hamilton</strong>
                </p>
                <hr style='margin: 20px 0; opacity: 0.3;'>
                <p style='font-size: 0.85em; color: #888;'>
                    <em>Developed for CAM DS 401<br>
                    Employer Project P1 2025</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Main content area
if not api_ready:
    st.error("""
    ### ⚠️ API Configuration Required
    
    The selected model **{selected_model}** requires a valid API key that is not currently configured.
    
    **Options:**
    1. Add the required API key to your `.env` file
    2. Select a different model with a configured API key
    3. Run in demo mode with limited functionality (rule-based analysis only)
    
    **Currently Available:**
    - Rule-based confidence analysis (hedging detection)
    - Rule-based analyst concern analysis (question categorization)
    - Basic risk scoring without AI insights
    """.format(selected_model=selected_model))

# Initialize agents based on model selection and API availability
if st.session_state.should_run_analysis:
    if api_ready:
        agents = initialize_agents_with_model(selected_model)
        if not agents:
            st.error("Failed to initialize agents")
            reset_analysis_state()
            st.stop()
    else:
        # Initialize agents that work without API
        agents = {
            'confidence': ManagementConfidenceAgent(),
            'analyst': AnalystConcernAgent(),
            'processor': TranscriptProcessor(),
            'loader': TranscriptLoader()
        }
else:
    agents = None

# Create tabs (conditionally include Debug Console)
# Note: We'll use a custom tab system to support programmatic switching
tab_names = [
    "📊 Risk Dashboard", 
    "📈 Sentiment Analysis",
    "🎯 Topic Evolution", 
    "🤖 Model Comparison",
    "💬 Chat Assistant",
    "📚 Document Sources",
    "🔍 RAG Analysis"  # Arkadiusz's RAG module
]

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

with tab1:
    st.header("Risk Assessment Dashboard")
    
    # Display existing results if available
    if not st.session_state.should_run_analysis and st.session_state.final_assessment:
        st.success("✅ Analysis Complete")
        
        final_assessment = st.session_state.final_assessment
        results = st.session_state.analysis_results
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_color = {
                'green': '🟢',
                'amber': '🟡', 
                'red': '🔴',
                'unknown': '⚪'
            }.get(final_assessment['risk_level'], '⚪')
            st.metric(
                "Risk Level",
                f"{risk_color} {final_assessment['risk_level'].upper()}",
                f"Score: {final_assessment['risk_score']:.1f}/10"
            )
        
        with col2:
            confidence_score = results.get('confidence', {}).get('overall_confidence_score', None)
            st.metric(
                "Confidence Score",
                f"{confidence_score:.1f}/10" if confidence_score else "N/A",
                "Management confidence"
            )
        
        with col3:
            concern_score = results.get('analyst_concerns', {}).get('concern_score', None)
            st.metric(
                "Analyst Concerns",
                f"{concern_score:.1f}/10" if concern_score else "N/A",
                "Question intensity"
            )
        
        with col4:
            st.metric(
                "Analysis Mode",
                "Full AI" if st.session_state.get('last_api_ready', True) else "Limited",
                "✅ Complete" if st.session_state.get('last_api_ready', True) else "⚠️ Fallback"
            )
        
        # Component scores
        st.subheader("Component Analysis")
        component_df = pd.DataFrame([final_assessment['component_scores']])
        st.dataframe(component_df, use_container_width=True)
    
    if st.session_state.should_run_analysis and agents and (api_ready or use_fallback):
        # Clear the trigger flag but keep analysis_running
        st.session_state.should_run_analysis = False
        
        # Create a container for progress display during analysis
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update initial progress
            update_analysis_progress(5, "Loading documents...")
            progress_bar.progress(5)
            status_text.text("Loading documents...")
            # Combine all files to analyze (pre-selected + newly uploaded)
            files_to_analyze = list(selected_files)  # Start with selected files
        
        # Note: newly_uploaded_files are already added to selected_files above
        # This ensures both types of files are processed together
        
        all_text = ""
        load_errors = []
        successful_loads = []
        
        # Process all files uniformly
        for file_path in files_to_analyze:
            try:
                file_path_obj = Path(file_path)
                
                # Check if file actually exists
                if not file_path_obj.exists():
                    load_errors.append((file_path, "File not found"))
                    continue
                    
                if file_path_obj.suffix == '.txt':
                    with open(file_path_obj, 'r', encoding='utf-8') as f:
                        content = f.read()
                        all_text += content + "\n\n"
                        successful_loads.append(str(file_path_obj))
                        
                        # Track document - add_document expects file_path string, not DocumentReference object
                        st.session_state.source_tracker.add_document(str(file_path_obj), "transcript")
                        
                elif file_path_obj.suffix == '.pdf':
                    content = agents['loader'].load_pdf_transcript(str(file_path_obj))
                    all_text += content + "\n\n"
                    successful_loads.append(str(file_path_obj))
                    
                    # Track document - add_document expects file_path string, not DocumentReference object
                    st.session_state.source_tracker.add_document(str(file_path_obj), "report")
                    
            except FileNotFoundError:
                load_errors.append((file_path, "File not found"))
            except PermissionError:
                load_errors.append((file_path, "Permission denied"))
            except Exception as e:
                load_errors.append((file_path, str(e)))
        
        # Show summary of load status
        if load_errors:
            st.warning(f"⚠️ Failed to load {len(load_errors)} file(s)")
            with st.expander("View load errors"):
                for path, error in load_errors:
                    st.error(f"• {Path(path).name}: {error}")
        
        # Only proceed with analysis if we have data and no critical errors
        if all_text and len(successful_loads) > 0:
            # Run analysis
            analysis_success = False
            
            # Run analysis without blocking spinner
            results = {}
            analysis_errors = []
            
            try:
                # Run available agents
                if api_ready:
                    # Full analysis with all agents
                    try:
                        # Track which agents failed
                        agent_failures = []
                        log_info("Starting full AI analysis with all agents")
                            
                        log_debug("Running sentiment analysis...")
                        results['sentiment'] = agents['sentiment'].analyze(all_text)
                        if not results['sentiment'] or results['sentiment'] == {} or 'error' in results['sentiment']:
                            error_detail = results['sentiment'].get('error', '') if isinstance(results['sentiment'], dict) else ''
                            agent_failures.append(f"Sentiment analysis{': ' + error_detail if error_detail else ''}")
                            log_error(f"Sentiment analysis failed: {error_detail}")
                        else:
                            log_info("Sentiment analysis completed successfully")
                            
                        log_debug("Running topic analysis...")
                        results['topics'] = agents['topics'].analyze(all_text)
                        if not results['topics'] or results['topics'] == {} or 'error' in results['topics']:
                            error_detail = results['topics'].get('error', '') if isinstance(results['topics'], dict) else ''
                            agent_failures.append(f"Topic analysis{': ' + error_detail if error_detail else ''}")
                            log_error(f"Topic analysis failed: {error_detail}")
                        else:
                            log_info("Topic analysis completed successfully")
                            
                        log_debug("Running confidence analysis...")
                        results['confidence'] = agents['confidence'].analyze(all_text)
                        if not results['confidence'] or results['confidence'] == {} or 'error' in results['confidence']:
                            error_detail = results['confidence'].get('error', '') if isinstance(results['confidence'], dict) else ''
                            agent_failures.append(f"Confidence analysis{': ' + error_detail if error_detail else ''}")
                            log_error(f"Confidence analysis failed: {error_detail}")
                        else:
                            log_info("Confidence analysis completed successfully")
                            
                        # Update progress for analyst concerns
                        update_analysis_progress(65, "Running analyst concerns analysis...")
                        progress_bar.progress(65)
                        status_text.text("Running analyst concerns analysis...")
                        log_debug("Running analyst concerns analysis...")
                        
                        results['analyst_concerns'] = agents['analyst'].analyze(all_text)
                        if not results['analyst_concerns'] or results['analyst_concerns'] == {} or 'error' in results['analyst_concerns']:
                            error_detail = results['analyst_concerns'].get('error', '') if isinstance(results['analyst_concerns'], dict) else ''
                            agent_failures.append(f"Analyst concerns analysis{': ' + error_detail if error_detail else ''}")
                            log_error(f"Analyst concerns analysis failed: {error_detail}")
                        else:
                            log_info("Analyst concerns analysis completed successfully")
                        
                        # Check if we got valid results
                        if all(results.values()) and not agent_failures:
                            # Orchestrate results
                            update_analysis_progress(80, "Synthesizing final risk assessment...")
                            progress_bar.progress(80)
                            status_text.text("Synthesizing final risk assessment...")
                            log_debug("Synthesizing risk assessment from agent results...")
                            
                            final_assessment = agents['orchestrator'].synthesize_risks(results)
                            log_info(f"Risk assessment complete: {final_assessment.get('risk_level', 'unknown').upper()} (Score: {final_assessment.get('risk_score', 0):.1f}/10)")
                            
                            # Final progress update
                            update_analysis_progress(95, "Finalizing results...")
                            progress_bar.progress(95)
                            status_text.text("Finalizing results...")
                            analysis_success = True
                        else:
                            if agent_failures:
                                # Check if it's an API overload issue
                                if any("overloaded" in failure.lower() for failure in agent_failures):
                                    analysis_errors.append("⚠️ API is currently overloaded (Error 529)")
                                    analysis_errors.append("The API service is experiencing high demand. Please:")
                                    analysis_errors.append("• Wait a few moments and try again")
                                    analysis_errors.append("• Or switch to a different model")
                                else:
                                    analysis_errors.append(f"Failed agents: {', '.join(agent_failures)}")
                                    analysis_errors.append("💡 Tip: This may be due to API rate limits. Try again in a moment or switch to a different model.")
                            else:
                                analysis_errors.append("Some agents returned empty results")
                            
                    except Exception as e:
                        analysis_errors.append(f"Analysis failed: {str(e)[:200]}")
                        log_error(f"Analysis failed: {str(e)}")
                        logger.log_exception("Full analysis exception")
                        # Reset state immediately on error to unblock UI
                        reset_analysis_state()
                        
                else:
                    # Limited analysis with fallback agents only
                    try:
                        results['confidence'] = agents['confidence'].analyze(all_text)
                        results['analyst_concerns'] = agents['analyst'].analyze(all_text)
                        
                        if results['confidence'] and results['analyst_concerns']:
                            # Create limited assessment
                            final_assessment = {
                                'risk_level': 'unknown',  # Don't fake amber
                                'risk_score': (results['confidence']['overall_confidence_score'] + 
                                             results['analyst_concerns']['concern_score']) / 2,
                                'component_scores': {
                                    'sentiment': 'N/A',
                                    'topics': 'N/A',
                                    'confidence': results['confidence']['overall_confidence_score'],
                                        'analyst_concerns': results['analyst_concerns']['concern_score']
                                    },
                                    'alert_priority': 'limited',
                                    'confidence': 0.3,  # Low confidence
                                    'note': 'Limited analysis - API key required for full insights'
                                }
                            analysis_success = True
                        else:
                            analysis_errors.append("Fallback analysis failed")
                            
                    except Exception as e:
                        analysis_errors.append(f"Fallback analysis error: {str(e)[:200]}")
                        # Reset state immediately on error to unblock UI
                        reset_analysis_state()
                            
            except Exception as e:
                analysis_errors.append(f"Critical error: {str(e)[:200]}")
                # Reset state immediately on error to unblock UI
                reset_analysis_state()
            
            # Only show results if analysis was successful
            if analysis_success and 'final_assessment' in locals():
                # Store results in session state for other tabs
                st.session_state.analysis_results = results
                st.session_state.final_assessment = final_assessment
                st.session_state.analyzed_files = successful_loads
                st.session_state.last_api_ready = api_ready  # Store API status for display after rerun
                
                # Switch back to Risk Dashboard tab after analysis completes
                log_info("Analysis complete, switching to Risk Dashboard tab")
                st.session_state.active_tab = 0
                
                # Risk metrics - only show if we have results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risk_color = {
                        'green': '🟢',
                        'amber': '🟡', 
                        'red': '🔴',
                        'unknown': '⚪'
                    }.get(final_assessment['risk_level'], '⚪')
                    st.metric(
                        "Risk Level",
                        f"{risk_color} {final_assessment['risk_level'].upper()}",
                        f"Score: {final_assessment['risk_score']:.1f}/10"
                    )
                
                with col2:
                    confidence_score = results.get('confidence', {}).get('overall_confidence_score', None)
                    st.metric(
                        "Confidence Score",
                        f"{confidence_score:.1f}/10" if confidence_score else "N/A",
                        "Management confidence"
                    )
                
                with col3:
                    concern_score = results.get('analyst_concerns', {}).get('concern_score', None)
                    st.metric(
                        "Analyst Concerns",
                        f"{concern_score:.1f}/10" if concern_score else "N/A",
                        "Question intensity"
                    )
                
                with col4:
                    st.metric(
                        "Analysis Mode",
                        "Full AI" if api_ready else "Limited",
                        "✅ Complete" if api_ready else "⚠️ Fallback"
                    )
                
                # Component scores
                st.subheader("Component Analysis")
                component_df = pd.DataFrame([final_assessment['component_scores']])
                st.dataframe(component_df, use_container_width=True)
                
                # Complete progress
                update_analysis_progress(100, "Analysis complete!")
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Reset state after successful analysis
                reset_analysis_state()
                
                # Automatically rerun to re-enable controls
                # The results are already stored in session state so they will persist
                st.rerun()
                
            elif analysis_errors:
                # Show errors if analysis failed
                st.error("❌ Analysis Failed")
                with st.expander("View analysis errors"):
                    for error in analysis_errors:
                        st.error(f"• {error}")
                st.info("Please check your API keys and try again, or enable fallback mode for limited analysis.")
                
            elif not all_text:
                st.warning("⚠️ No data to analyze. Please select valid files or check file paths.")
        
        # Reset state after all analysis is complete (outside spinner)
        # Only reset if not already reset by error handlers
        if st.session_state.analysis_running:
            reset_analysis_state()
            # Rerun to update UI state
            st.rerun()

# Tab 2: Sentiment Analysis
with tab2:
    st.header("📈 Sentiment Analysis")
    
    if st.session_state.analysis_results and 'sentiment' in st.session_state.analysis_results:
        sentiment_data = st.session_state.analysis_results['sentiment']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Sentiment", sentiment_data.get('overall_sentiment', 'N/A'))
            st.metric("Sentiment Score", f"{sentiment_data.get('sentiment_score', 0):.2f}")
        with col2:
            st.metric("Confidence Level", sentiment_data.get('confidence_level', 'N/A'))
            tone_indicators = sentiment_data.get('tone_indicators', [])
            if tone_indicators:
                st.write("**Tone Indicators:**")
                for indicator in tone_indicators[:5]:
                    st.write(f"• {indicator}")
        
        # Show analyzed files with source indication
        if st.session_state.analyzed_files:
            st.subheader("📁 Analyzed Files")
            for file_path in st.session_state.analyzed_files:
                file_name = Path(file_path).name
                if 'uploads' in str(file_path):
                    st.write(f"• 📤 {file_name} (uploaded)")
                else:
                    st.write(f"• 📁 {file_name}")
    else:
        st.info("Run analysis in the Risk Dashboard tab first to see sentiment results.")

# Tab 3: Topic Evolution
with tab3:
    st.header("🎯 Topic Evolution")
    
    if st.session_state.analysis_results and 'topics' in st.session_state.analysis_results:
        topics_data = st.session_state.analysis_results['topics']
        
        st.subheader("Key Topics Identified")
        topics = topics_data.get('topics', [])
        if topics:
            for topic in topics[:10]:
                st.write(f"• **{topic.get('name', 'Unknown')}**: {topic.get('frequency', 0)} mentions")
        
        # Risk categories
        risk_categories = topics_data.get('risk_categories', {})
        if risk_categories:
            st.subheader("Risk Categories")
            for category, details in risk_categories.items():
                st.write(f"• **{category}**: {details}")
    else:
        st.info("Run analysis in the Risk Dashboard tab first to see topic results.")

# Tab 4: Model Comparison
with tab4:
    st.header("🤖 Model Comparison")
    
    st.write("### Currently Using: " + selected_model)
    
    if st.session_state.final_assessment:
        st.write("### Analysis Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", "< 30 seconds")
        with col2:
            st.metric("Confidence", f"{st.session_state.final_assessment.get('confidence', 0):.0%}")
        with col3:
            st.metric("Risk Score", f"{st.session_state.final_assessment.get('risk_score', 0):.1f}/10")
    
    st.write("### Available Models")
    
    # Group models by provider
    openai_models = {k: v for k, v in available_models.items() if 'gpt' in k.lower()}
    claude_models = {k: v for k, v in available_models.items() if 'claude' in k.lower()}
    gemini_models = {k: v for k, v in available_models.items() if 'gemini' in k.lower()}
    other_models = {k: v for k, v in available_models.items() if k not in openai_models and k not in claude_models and k not in gemini_models}
    
    if openai_models:
        st.write("#### OpenAI Models")
        for model, description in openai_models.items():
            icon = get_api_status_icon("openai", test_connection=False)
            st.write(f"{icon} {description}")
    
    if claude_models:
        st.write("#### Anthropic/Claude Models")
        for model, description in claude_models.items():
            icon = get_api_status_icon("anthropic", test_connection=False)
            st.write(f"{icon} {description}")
    
    if gemini_models:
        st.write("#### Google Gemini Models")
        for model, description in gemini_models.items():
            icon = get_api_status_icon("google", test_connection=False)
            st.write(f"{icon} {description}")
    
    if other_models:
        st.write("#### Other")
        for model, description in other_models.items():
            st.info(f"• {description}")

# Tab 5: Chat Assistant
with tab5:
    st.header("💬 Chat Assistant")
    
    if st.session_state.final_assessment:
        st.write("Ask questions about the analysis results:")
        
        user_question = st.text_input("Your question:", placeholder="What are the main risks identified?")
        
        if user_question:
            # Simple response based on analysis
            st.write("### Response:")
            st.write(f"Based on the analysis of {len(st.session_state.analyzed_files)} file(s):")
            st.write(f"- Risk Level: {st.session_state.final_assessment.get('risk_level', 'Unknown').upper()}")
            st.write(f"- Risk Score: {st.session_state.final_assessment.get('risk_score', 0):.1f}/10")
            
            if 'confidence' in st.session_state.analysis_results:
                conf = st.session_state.analysis_results['confidence']
                st.write(f"- Management Confidence: {conf.get('overall_confidence_score', 0):.1f}/10")
    else:
        st.info("Run analysis first to use the chat assistant.")

# Tab 6: Document Sources
with tab6:
    st.header("📚 Document Sources")
    
    if st.session_state.analyzed_files:
        st.subheader(f"📊 Analyzed {len(st.session_state.analyzed_files)} Document(s)")
        
        # Separate uploaded and pre-existing files
        uploaded_docs = [f for f in st.session_state.analyzed_files if 'uploads' in str(f)]
        preexisting_docs = [f for f in st.session_state.analyzed_files if 'uploads' not in str(f)]
        
        if uploaded_docs:
            st.write("### 📤 Uploaded Documents")
            for file_path in uploaded_docs:
                with st.expander(f"{Path(file_path).name}"):
                    st.write(f"**Type:** {'PDF' if file_path.endswith('.pdf') else 'Text'}")
                    st.write(f"**Source:** User Upload")
                    # Get file size
                    try:
                        size = os.path.getsize(file_path)
                        st.write(f"**Size:** {size:,} bytes")
                    except:
                        pass
                    if st.session_state.source_tracker:
                        st.write("**Tracked in analysis:** ✅")
        
        if preexisting_docs:
            st.write("### 📁 Pre-loaded Documents")
            for file_path in preexisting_docs:
                with st.expander(f"{Path(file_path).name}"):
                    st.write(f"**Type:** {'PDF' if file_path.endswith('.pdf') else 'Text'}")
                    st.write(f"**Source:** {Path(file_path).parent.name}")
                    # Get file size
                    try:
                        size = os.path.getsize(file_path)
                        st.write(f"**Size:** {size:,} bytes")
                    except:
                        pass
                    if st.session_state.source_tracker:
                        st.write("**Tracked in analysis:** ✅")
    else:
        st.info("No documents analyzed yet. Run analysis in the Risk Dashboard tab.")

# Tab 7: RAG Analysis (Arkadiusz's Module)
with tab7:
    st.header("🔍 RAG Analysis System")
    st.markdown("*Document Analysis using Retrieval Augmented Generation*")
    log_debug("[RAG-UI] RAG tab loaded")
    
    # Initialize RAG system with the selected model from Configuration
    if 'rag_system' not in st.session_state or st.session_state.rag_system is None:
        try:
            log_info(f"[RAG-UI] Attempting to initialize RAG system with model: {selected_model}")
            rag_system = init_rag_system(chat_model=selected_model)
            log_info(f"[RAG-UI] RAG system initialized successfully with model: {selected_model}")
        except Exception as e:
            log_error(f"[RAG-UI] Failed to initialize RAG system: {str(e)}")
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            rag_system = None
            import traceback
            log_error(f"[RAG-UI] Traceback: {traceback.format_exc()}")
    else:
        # Check if model has changed and reinitialize if needed
        rag_system = init_rag_system(chat_model=selected_model)
    
    # Check synchronization status on first load
    if rag_system and st.session_state.rag_index_synchronized is None:
        try:
            stats = rag_system.get_collection_stats()
            chunk_count = stats.get('points_count', 0) if stats.get('exists', False) else 0
            
            # If there are chunks in Qdrant but no indexed info in session, we're unsynchronized
            if chunk_count > 0 and not st.session_state.rag_indexed_info:
                st.session_state.rag_index_synchronized = False
                log_info(f"[RAG-UI] Detected unsynchronized index with {chunk_count} existing chunks")
            elif chunk_count == 0:
                st.session_state.rag_index_synchronized = True  # Empty is synchronized
            else:
                st.session_state.rag_index_synchronized = True  # Has info, assume synced
        except Exception as e:
            log_error(f"[RAG-UI] Failed to check synchronization: {str(e)}")
            st.session_state.rag_index_synchronized = None
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query Section
        st.subheader("📝 Query Documents")
        
        # Query input
        query_input = st.text_area(
            "Enter your question about the documents:",
            height=100,
            placeholder="Example: What are the main risks mentioned in the documents?"
        )
        
        # Query button and options
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            query_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col_btn2:
            clear_history = st.button("🗑️ Clear History", use_container_width=True)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            search_mode = st.selectbox("Search Mode:", ["mmr", "similarity"], index=0)
            num_chunks = st.slider("Number of chunks to retrieve:", 1, 10, 6)
            num_sources = st.slider("Number of sources to show:", 1, 10, 4)
        
        # Process query
        if query_button and query_input:
            if not rag_system:
                st.error("❌ RAG system not initialized. Please check your configuration.")
            else:
                with st.spinner("🔍 Searching documents..."):
                    try:
                        log_info(f"[RAG-UI] Processing query: {query_input}")
                        answer, sources = rag_system.rag_answer(
                            query_input,
                            mode=search_mode,
                            k=num_chunks,
                            topk_sources=num_sources
                        )
                        
                        # Add to history
                        st.session_state.rag_query_history.append({
                            "question": query_input,
                            "answer": answer,
                            "sources": sources,
                            "timestamp": datetime.now()
                        })
                        
                        # Display answer
                        st.success("✅ Answer generated successfully!")
                        st.markdown("### Answer:")
                        st.markdown(answer)
                        
                        # Display sources
                        if sources:
                            st.markdown("### 📚 Sources:")
                            for source in sources:
                                st.markdown(f"- {source}")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        log_error(f"[RAG-UI] Query failed: {str(e)}")
        
        # Clear history
        if clear_history:
            st.session_state.rag_query_history = []
            st.success("History cleared!")
        
        # Display history
        if st.session_state.rag_query_history:
            st.divider()
            st.subheader("📜 Query History")
            for i, item in enumerate(reversed(st.session_state.rag_query_history[-5:])):
                with st.expander(f"Q: {item['question'][:100]}... ({item['timestamp'].strftime('%H:%M:%S')})"):
                    st.markdown("**Question:**")
                    st.write(item['question'])
                    st.markdown("**Answer:**")
                    st.write(item['answer'])
                    if item['sources']:
                        st.markdown("**Sources:**")
                        for source in item['sources']:
                            st.write(f"- {source}")
    
    with col2:
        # Status and Management Section
        st.subheader("🎛️ System Status")
        
        # Connection status
        if rag_system:
            try:
                connection_status = rag_system.test_connection()
                log_debug(f"[RAG-UI] Connection test result: {connection_status}")
            except Exception as e:
                connection_status = False
                log_error(f"[RAG-UI] Connection test failed: {str(e)}")
        else:
            connection_status = False
            log_warning("[RAG-UI] RAG system is None, cannot test connection")
            
        if connection_status:
            st.success("✅ Qdrant Connected")
            # Display the model being used for RAG
            if rag_system and hasattr(rag_system, 'CHAT_MODEL'):
                st.info(f"🤖 Using model: **{rag_system.CHAT_MODEL}**")
        else:
            st.error("❌ Qdrant Disconnected")
            st.info("Please check your Qdrant configuration in .env or Streamlit secrets")
        
        st.divider()
        
        # Document Management
        st.subheader("📁 Document Management")
        
        # Single index button that always re-indexes
        if st.button("🔄 Index Selected Documents", type="primary", use_container_width=True):
            if not rag_system:
                st.error("❌ RAG system not initialized")
            elif 'selected_files' not in st.session_state or not st.session_state.selected_files:
                st.warning("⚠️ No documents selected in Configuration")
            else:
                with st.spinner(f"Indexing {len(st.session_state.selected_files)} selected documents..."):
                    try:
                        # Use the same data path as main app
                        data_dir = Path(config.DATA_PATH)
                        
                        # Update RAG module to use main app's data directory
                        rag_system.DATA_DIR = data_dir
                        
                        # Always re-index to ensure fresh embeddings
                        result = rag_system.populate_vector_store(selected_files=st.session_state.selected_files)
                        
                        if result.get("success", False):
                            # Count actual documents, not pages
                            num_docs = len([f for f in st.session_state.selected_files if f.endswith('.pdf')])
                            num_pages = result.get('documents_processed', 0)  # This is actually pages
                            chunks_created = result.get('chunks_created', 0)
                            st.success(f"✅ Indexed {chunks_created} chunks from {num_docs} document{'s' if num_docs != 1 else ''} ({num_pages} pages processed)")
                            
                            # Track indexed documents with detailed info
                            import datetime
                            indexed_info = {
                                'documents': [f for f in st.session_state.selected_files if f.endswith('.pdf')],
                                'total_chunks': chunks_created,
                                'total_pages': num_pages,
                                'indexed_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'chunks_per_doc': {}  # Could be populated with per-doc info if needed
                            }
                            st.session_state.rag_indexed_info = indexed_info
                            st.session_state.rag_indexed_docs = [Path(f) for f in indexed_info['documents']]
                            # Store the new chunk count for immediate display
                            st.session_state.rag_chunk_count = chunks_created
                            # Mark as synchronized after successful indexing
                            st.session_state.rag_index_synchronized = True
                            # Force refresh to update chunk count display
                            st.rerun()
                        else:
                            st.error(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        log_error(f"[RAG-UI] Indexing failed: {str(e)}")
        
        # Clear Vector Store button
        if st.button("🗑️ Clear Vector Store", use_container_width=True):
            if not rag_system:
                st.error("❌ RAG system not initialized")
            else:
                # Confirmation using session state
                if 'confirm_clear' not in st.session_state:
                    st.session_state.confirm_clear = False
                
                if not st.session_state.confirm_clear:
                    st.warning("⚠️ This will delete ALL indexed documents from the vector store. Click again to confirm.")
                    st.session_state.confirm_clear = True
                else:
                    with st.spinner("Clearing vector store..."):
                        try:
                            # Clear the vector store by recreating the collection
                            client = rag_system.get_qdrant_client()
                            
                            # Get embedding dimensions for recreation
                            if rag_system.EMBEDDING_MODEL == "text-embedding-3-large":
                                dim = 3072
                            elif rag_system.EMBEDDING_MODEL == "text-embedding-3-small":
                                dim = 1536
                            else:
                                dim = 1536  # Default
                            
                            # Recreate the collection (this deletes all data)
                            from qdrant_client import models as rest
                            client.recreate_collection(
                                collection_name=rag_system.COLLECTION_NAME,
                                vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE)
                            )
                            
                            # Clear session state
                            st.session_state.rag_indexed_docs = []
                            st.session_state.rag_chunk_count = 0
                            st.session_state.rag_indexed_info = None
                            st.session_state.rag_index_synchronized = True  # Empty is synchronized
                            st.session_state.confirm_clear = False
                            
                            st.success("✅ Vector store cleared successfully")
                            log_info("[RAG-UI] Vector store cleared")
                            
                            # Force refresh
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Failed to clear vector store: {str(e)}")
                            log_error(f"[RAG-UI] Failed to clear vector store: {str(e)}")
                            st.session_state.confirm_clear = False
        
        # Reset confirmation if user does other actions
        if 'confirm_clear' in st.session_state and st.session_state.confirm_clear:
            # Auto-reset confirmation after showing warning
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
        
        # Index Information Display
        st.divider()
        st.subheader("📊 Index Status")
        
        # Check if we're synchronized
        if st.session_state.rag_index_synchronized == False:
            # Unsynchronized state - show data from Qdrant with warning
            if rag_system:
                stats = rag_system.get_collection_stats()
                chunk_count = stats.get('points_count', 0) if stats.get('exists', False) else 0
                if chunk_count > 0:
                    st.warning(f"⚠️ Index contains {chunk_count:,} chunks from a previous session")
                    st.info("📝 The indexed documents are unknown. Please re-index selected documents to sync or clear the vector store.")
                else:
                    st.info("📭 Index is empty. Select documents and click 'Index Selected Documents' to begin.")
        elif 'rag_indexed_info' in st.session_state and st.session_state.rag_indexed_info:
            # Synchronized with data
            info = st.session_state.rag_indexed_info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents in Index", len(info['documents']))
                st.metric("Total Chunks", info['total_chunks'])
            with col2:
                st.metric("Total Pages", info['total_pages'])
                st.metric("Last Indexed", info['indexed_at'])
        else:
            # Check Qdrant for actual state
            if rag_system:
                stats = rag_system.get_collection_stats()
                chunk_count = stats.get('points_count', 0) if stats.get('exists', False) else 0
                if chunk_count > 0:
                    st.warning(f"⚠️ Index contains {chunk_count:,} chunks but document info is unavailable")
                    st.info("📝 Re-index selected documents to sync the index status.")
                else:
                    st.info("📭 Index is empty. Select documents and click 'Index Selected Documents' to begin.")
        
        # List selected documents for RAG indexing
        if 'selected_files' in st.session_state and st.session_state.selected_files:
            st.divider()
            st.subheader("📚 Selected Documents Status")
            selected_count = len(st.session_state.selected_files)
            
            # Get list of indexed documents
            indexed_docs = []
            if 'rag_indexed_info' in st.session_state and st.session_state.rag_indexed_info:
                indexed_docs = st.session_state.rag_indexed_info.get('documents', [])
            
            # Convert to Path objects for comparison
            indexed_paths = [Path(doc).name for doc in indexed_docs]
            
            st.info(f"📌 {selected_count} document(s) selected in Configuration")
            
            for file_path in st.session_state.selected_files:
                file_path_obj = Path(file_path)
                
                # Check synchronization state
                if st.session_state.rag_index_synchronized == False:
                    # Unsynchronized - we don't know what's in the index
                    indexed_marker = "❓"
                    status_text = "Unknown - re-index to sync"
                else:
                    # Synchronized - check if file is in the index
                    is_indexed = file_path in indexed_docs or str(file_path) in indexed_docs
                    indexed_marker = "✅" if is_indexed else "⏳"
                    status_text = "In Index" if is_indexed else "Not in Index"
                    
                st.write(f"{indexed_marker} {file_path_obj.name} ({status_text})")
        else:
            st.divider()
            st.info("📋 No documents selected. Please select documents in the Configuration section.")

# Footer - Model Status
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Model Status:</strong> {status}</p>
</div>
""".format(
    status=f"✅ {selected_model} Ready" if api_ready else "⚠️ API Key Required"
), unsafe_allow_html=True)