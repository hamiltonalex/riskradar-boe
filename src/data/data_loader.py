"""
Data Loader for Earning Call Transcripts
Handles loading transcripts from various sources
"""

import os
import json
import requests
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import PyPDF2
import pdfplumber
# Import logging functions
try:
    from src.utils.debug_logger import log_debug, log_info, log_error, log_warning
except ImportError:
    # Fallback if logger not available
    def log_debug(msg): print(f"[DEBUG] {msg}")
    def log_info(msg): print(f"[INFO] {msg}")
    def log_error(msg): print(f"[ERROR] {msg}")
    def log_warning(msg): print(f"[WARNING] {msg}")

class TranscriptLoader:
    """Load earning call transcripts from multiple sources"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.transcripts_path = self.data_path / "transcripts"
        self.transcripts_path.mkdir(parents=True, exist_ok=True)
        
        # Sample transcript URLs (would be replaced with actual sources)
        self.sample_sources = {
            'JPM': {
                'company': 'JP Morgan Chase',
                'ticker': 'JPM',
                'transcripts': []
            },
            'HSBC': {
                'company': 'HSBC Holdings',
                'ticker': 'HSBC',
                'transcripts': []
            },
            'C': {
                'company': 'Citigroup',
                'ticker': 'C',
                'transcripts': []
            }
        }
    
    def load_pdf_transcript(self, pdf_path: str) -> str:
        """Extract text from PDF transcript"""
        text = ""
        
        try:
            # Try pdfplumber first (better for complex PDFs)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
            except Exception as e:
                log_error(f"Error reading PDF {pdf_path}: {e}")
                return ""
        
        return text
    
    def load_text_transcript(self, text_path: str) -> str:
        """Load text transcript from file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            log_error(f"Error reading text file {text_path}: {e}")
            return ""
    
    def load_json_transcript(self, json_path: str) -> Dict:
        """Load structured transcript from JSON"""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            log_error(f"Error reading JSON file {json_path}: {e}")
            return {}
    
    def load_from_directory(self, directory: str) -> List[Dict]:
        """Load all transcripts from a directory"""
        transcripts = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            log_warning(f"Directory {directory} does not exist")
            return transcripts
        
        # Process different file types
        for file_path in directory_path.iterdir():
            if file_path.suffix == '.pdf':
                text = self.load_pdf_transcript(str(file_path))
                if text:
                    transcripts.append({
                        'filename': file_path.name,
                        'type': 'pdf',
                        'content': text
                    })
            elif file_path.suffix in ['.txt', '.md']:
                text = self.load_text_transcript(str(file_path))
                if text:
                    transcripts.append({
                        'filename': file_path.name,
                        'type': 'text',
                        'content': text
                    })
            elif file_path.suffix == '.json':
                data = self.load_json_transcript(str(file_path))
                if data:
                    transcripts.append({
                        'filename': file_path.name,
                        'type': 'json',
                        'content': data
                    })
        
        return transcripts
    
    def load_historical_transcripts(self, ticker: str, start_date: str, end_date: str) -> List[Dict]:
        """Load historical transcripts for a specific ticker within date range"""
        transcripts = []
        
        # Convert dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Look for transcripts in the data directory
        pattern = f"{ticker}_*_transcript.*"
        for file_path in self.transcripts_path.glob(pattern):
            # Extract date from filename (if available)
            # Add to list if within date range
            transcripts.append({
                'ticker': ticker,
                'file': str(file_path),
                'content': self.load_text_transcript(str(file_path))
            })
        
        return transcripts