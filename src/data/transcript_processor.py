"""
Earning Call Transcript Processor
Handles ingestion, parsing, and segmentation of earning call transcripts
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from pathlib import Path

@dataclass
class TranscriptSegment:
    """Represents a segment of an earning call transcript"""
    speaker: str
    role: Optional[str]
    content: str
    segment_type: str  # 'presentation', 'qa', 'closing'
    timestamp: Optional[str] = None
    
@dataclass
class EarningCallTranscript:
    """Complete earning call transcript with metadata"""
    company: str
    date: datetime
    quarter: str
    year: int
    segments: List[TranscriptSegment]
    metadata: Dict
    raw_text: str

class TranscriptProcessor:
    """Process and parse earning call transcripts"""
    
    def __init__(self):
        self.executive_titles = [
            'CEO', 'CFO', 'COO', 'President', 'Chairman',
            'Chief Executive', 'Chief Financial', 'Chief Operating',
            'Vice President', 'Director', 'Head of', 'Managing Director'
        ]
        
        self.analyst_patterns = [
            r'analyst', r'research', r'capital', r'securities',
            r'bank', r'partners', r'asset', r'management'
        ]
        
    def load_transcript(self, filepath: str) -> str:
        """Load transcript from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse_transcript(self, text: str, company: str, date: str) -> EarningCallTranscript:
        """Parse raw transcript text into structured format"""
        
        # Extract metadata
        metadata = self._extract_metadata(text)
        
        # Parse date
        transcript_date = datetime.strptime(date, '%Y-%m-%d')
        quarter = self._determine_quarter(transcript_date)
        
        # Segment the transcript
        segments = self._segment_transcript(text)
        
        return EarningCallTranscript(
            company=company,
            date=transcript_date,
            quarter=quarter,
            year=transcript_date.year,
            segments=segments,
            metadata=metadata,
            raw_text=text
        )
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata from transcript header"""
        metadata = {}
        
        # Extract ticker symbol
        ticker_match = re.search(r'\(([A-Z]{1,5})\)', text[:500])
        if ticker_match:
            metadata['ticker'] = ticker_match.group(1)
        
        # Extract call type
        if 'earnings' in text[:1000].lower():
            metadata['call_type'] = 'earnings'
        elif 'investor' in text[:1000].lower():
            metadata['call_type'] = 'investor_day'
        else:
            metadata['call_type'] = 'other'
            
        return metadata
    
    def _determine_quarter(self, date: datetime) -> str:
        """Determine fiscal quarter from date"""
        month = date.month
        if month <= 3:
            return 'Q1'
        elif month <= 6:
            return 'Q2'
        elif month <= 9:
            return 'Q3'
        else:
            return 'Q4'
    
    def _segment_transcript(self, text: str) -> List[TranscriptSegment]:
        """Segment transcript into speakers and sections"""
        segments = []
        
        # Split into presentation and Q&A sections
        qa_start = self._find_qa_section(text)
        
        if qa_start:
            presentation_text = text[:qa_start]
            qa_text = text[qa_start:]
            
            # Process presentation section
            segments.extend(self._parse_section(presentation_text, 'presentation'))
            
            # Process Q&A section
            segments.extend(self._parse_section(qa_text, 'qa'))
        else:
            # No clear Q&A section found
            segments.extend(self._parse_section(text, 'presentation'))
        
        return segments
    
    def _find_qa_section(self, text: str) -> Optional[int]:
        """Find the start of Q&A section"""
        qa_patterns = [
            r'question[- ]and[- ]answer',
            r'q\s*&\s*a\s*session',
            r'now\s+open\s+.*\s+questions',
            r'take\s+.*\s+questions',
            r'operator.*first\s+question'
        ]
        
        for pattern in qa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.start()
        
        return None
    
    def _parse_section(self, text: str, section_type: str) -> List[TranscriptSegment]:
        """Parse a section of the transcript"""
        segments = []
        
        # Pattern to identify speakers
        speaker_pattern = r'^([A-Z][A-Za-z\s\-\.]+)(?:\s*[-–]\s*([A-Za-z\s,]+))?:'
        
        lines = text.split('\n')
        current_speaker = None
        current_role = None
        current_content = []
        
        for line in lines:
            speaker_match = re.match(speaker_pattern, line)
            
            if speaker_match:
                # Save previous segment if exists
                if current_speaker and current_content:
                    segments.append(TranscriptSegment(
                        speaker=current_speaker,
                        role=current_role,
                        content=' '.join(current_content).strip(),
                        segment_type=section_type
                    ))
                
                # Start new segment
                current_speaker = speaker_match.group(1).strip()
                current_role = self._identify_role(
                    current_speaker,
                    speaker_match.group(2) if speaker_match.group(2) else ''
                )
                current_content = [line[speaker_match.end():].strip()]
            else:
                # Continue current segment
                if line.strip():
                    current_content.append(line.strip())
        
        # Save final segment
        if current_speaker and current_content:
            segments.append(TranscriptSegment(
                speaker=current_speaker,
                role=current_role,
                content=' '.join(current_content).strip(),
                segment_type=section_type
            ))
        
        return segments
    
    def _identify_role(self, speaker: str, role_text: str) -> Optional[str]:
        """Identify speaker's role"""
        combined = f"{speaker} {role_text}".lower()
        
        # Check for executive roles
        for title in self.executive_titles:
            if title.lower() in combined:
                return 'executive'
        
        # Check for analyst roles
        for pattern in self.analyst_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return 'analyst'
        
        # Check for operator
        if 'operator' in combined:
            return 'operator'
        
        return 'unknown'
    
    def extract_segments_by_role(self, transcript: EarningCallTranscript, role: str) -> List[str]:
        """Extract all segments by a specific role"""
        return [
            segment.content 
            for segment in transcript.segments 
            if segment.role == role
        ]
    
    def extract_qa_pairs(self, transcript: EarningCallTranscript) -> List[Tuple[str, str]]:
        """Extract question-answer pairs from Q&A section"""
        qa_pairs = []
        qa_segments = [s for s in transcript.segments if s.segment_type == 'qa']
        
        i = 0
        while i < len(qa_segments):
            # Look for analyst question
            if qa_segments[i].role == 'analyst':
                question = qa_segments[i].content
                
                # Find next executive response
                answer = None
                for j in range(i + 1, min(i + 5, len(qa_segments))):
                    if qa_segments[j].role == 'executive':
                        answer = qa_segments[j].content
                        break
                
                if answer:
                    qa_pairs.append((question, answer))
            
            i += 1
        
        return qa_pairs
    
    def save_processed_transcript(self, transcript: EarningCallTranscript, output_path: str):
        """Save processed transcript to JSON"""
        data = {
            'company': transcript.company,
            'date': transcript.date.isoformat(),
            'quarter': transcript.quarter,
            'year': transcript.year,
            'metadata': transcript.metadata,
            'segments': [
                {
                    'speaker': s.speaker,
                    'role': s.role,
                    'content': s.content,
                    'segment_type': s.segment_type
                }
                for s in transcript.segments
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_processed_transcript(self, filepath: str) -> EarningCallTranscript:
        """Load processed transcript from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        segments = [
            TranscriptSegment(**segment)
            for segment in data['segments']
        ]
        
        return EarningCallTranscript(
            company=data['company'],
            date=datetime.fromisoformat(data['date']),
            quarter=data['quarter'],
            year=data['year'],
            segments=segments,
            metadata=data['metadata'],
            raw_text=""  # Not stored in JSON
        )