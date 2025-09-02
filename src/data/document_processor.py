"""
Document Processor for RiskRadar
Handles document metadata extraction, content structuring, and section parsing
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

class DocumentProcessor:
    """Process and structure documents for multi-document analysis"""
    
    def __init__(self):
        # Common bank name variations for standardization
        self.bank_name_map = {
            'jpm': 'JPMorgan',
            'jpmorgan': 'JPMorgan',
            'jp morgan': 'JPMorgan',
            'jpmorgan chase': 'JPMorgan',
            'hsbc': 'HSBC',
            'barclays': 'Barclays',
            'lloyds': 'Lloyds',
            'lloyds banking group': 'Lloyds',
            'natwest': 'NatWest',
            'nat west': 'NatWest',
            'rbs': 'RBS',
            'royal bank of scotland': 'RBS',
            'santander': 'Santander',
            'santander uk': 'Santander',
            'standard chartered': 'Standard Chartered',
            'deutsche': 'Deutsche Bank',
            'deutsche bank': 'Deutsche Bank',
            'ubs': 'UBS',
            'credit suisse': 'Credit Suisse',
            'cs': 'Credit Suisse',
            'boa': 'Bank of America',
            'bank of america': 'Bank of America',
            'bofa': 'Bank of America',
            'wells fargo': 'Wells Fargo',
            'wf': 'Wells Fargo',
            'citi': 'Citigroup',
            'citigroup': 'Citigroup',
            'citibank': 'Citigroup',
            'goldman': 'Goldman Sachs',
            'goldman sachs': 'Goldman Sachs',
            'gs': 'Goldman Sachs',
            'morgan stanley': 'Morgan Stanley',
            'ms': 'Morgan Stanley'
        }
        
        # Quarter mapping
        self.quarter_map = {
            'q1': 'Q1', '1q': 'Q1', 'first quarter': 'Q1',
            'q2': 'Q2', '2q': 'Q2', 'second quarter': 'Q2',
            'q3': 'Q3', '3q': 'Q3', 'third quarter': 'Q3',
            'q4': 'Q4', '4q': 'Q4', 'fourth quarter': 'Q4',
            'fy': 'FY', 'full year': 'FY', 'annual': 'FY'
        }
    
    def extract_metadata(self, file_path: str) -> Dict:
        """
        Extract metadata from filename and path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted metadata
        """
        path = Path(file_path)
        filename = path.stem.lower()  # Remove extension and lowercase
        
        metadata = {
            'file_path': str(file_path),
            'filename': path.name,
            'extension': path.suffix,
            'folder': path.parent.name
        }
        
        # Extract bank name
        bank_name = self._extract_bank_name(filename)
        metadata['bank_name'] = bank_name
        
        # Extract period (quarter and year)
        period = self._extract_period(filename)
        metadata['period'] = period
        metadata['quarter'] = period.split()[0] if period else None
        metadata['year'] = period.split()[1] if period and len(period.split()) > 1 else None
        
        # Extract document type
        doc_type = self._extract_document_type(filename)
        metadata['document_type'] = doc_type
        
        # Generate unique document ID
        metadata['doc_id'] = f"{bank_name}_{period}_{doc_type}".replace(' ', '_')
        
        return metadata
    
    def _extract_bank_name(self, filename: str) -> str:
        """Extract and standardize bank name from filename"""
        # Normalize filename for matching (replace underscores with spaces)
        normalized = filename.replace('_', ' ').replace('-', ' ')
        
        # Try to match known bank names
        for variant, standard_name in self.bank_name_map.items():
            if variant in normalized:
                return standard_name
        
        # If no match, try to extract first word/segment
        # Common patterns: bankname_q1_2025, bankname-2025-q1
        parts = re.split(r'[-_\s]', filename)
        if parts:
            potential_bank = parts[0]
            # Check if it might be a bank name (not a date or quarter)
            if not re.match(r'\d{4}|q\d|[0-9]+', potential_bank):
                return potential_bank.title()
        
        return "Unknown Bank"
    
    def _extract_period(self, filename: str) -> str:
        """Extract time period from filename"""
        # Extract year (4 digits)
        year_match = re.search(r'20\d{2}', filename)
        year = year_match.group() if year_match else None
        
        # Extract quarter
        quarter = None
        for q_variant, q_standard in self.quarter_map.items():
            if q_variant in filename:
                quarter = q_standard
                break
        
        # Also check for patterns like "1q25", "q325", "3q2025"
        if not quarter:
            q_pattern = re.search(r'([1-4])q(\d{2,4})|q([1-4])[-_]?(\d{2,4})?', filename)
            if q_pattern:
                groups = q_pattern.groups()
                if groups[0]:  # Pattern like "1q25"
                    quarter = f"Q{groups[0]}"
                    if groups[1] and len(groups[1]) == 2:
                        year = f"20{groups[1]}"
                elif groups[2]:  # Pattern like "q3" or "q3_2025"
                    quarter = f"Q{groups[2]}"
                    if groups[3]:
                        year = f"20{groups[3]}" if len(groups[3]) == 2 else groups[3]
        
        # Construct period string
        if quarter and year:
            return f"{quarter} {year}"
        elif year:
            return f"FY {year}"
        elif quarter:
            return quarter
        
        # Try to extract date in format YYYYMMDD
        date_match = re.search(r'(\d{8})', filename)
        if date_match:
            date_str = date_match.group()
            try:
                date = datetime.strptime(date_str, '%Y%m%d')
                quarter = f"Q{(date.month - 1) // 3 + 1}"
                return f"{quarter} {date.year}"
            except:
                pass
        
        return "Unknown Period"
    
    def _extract_document_type(self, filename: str) -> str:
        """Extract document type from filename"""
        doc_types = {
            'transcript': 'earnings_transcript',
            'earnings': 'earnings_transcript',
            'call': 'earnings_call',
            'presentation': 'presentation',
            'slides': 'presentation',
            'report': 'report',
            'annual': 'annual_report',
            'quarterly': 'quarterly_report',
            'results': 'results',
            'statement': 'financial_statement'
        }
        
        for keyword, doc_type in doc_types.items():
            if keyword in filename:
                return doc_type
        
        return "document"
    
    def extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract different sections from document content
        
        Args:
            content: Full document text
            
        Returns:
            Dictionary with section names and content
        """
        sections = {}
        content_lower = content.lower()
        
        # Common section markers
        section_markers = {
            'management_remarks': [
                'prepared remarks', 'management discussion', 
                'opening remarks', 'ceo remarks', 'cfo remarks'
            ],
            'qa': [
                'question-and-answer', 'q&a session', 'q and a',
                'questions and answers', 'analyst questions'
            ],
            'financial_results': [
                'financial results', 'financial performance',
                'results of operations', 'financial highlights'
            ],
            'guidance': [
                'guidance', 'outlook', 'forward-looking',
                'expectations', 'forecast'
            ]
        }
        
        # Try to find and extract sections
        for section_name, markers in section_markers.items():
            for marker in markers:
                pattern = re.compile(
                    rf'{re.escape(marker)}.*?(?=(?:' + 
                    '|'.join(re.escape(m) for markers_list in section_markers.values() 
                            for m in markers_list if m != marker) + 
                    r'|\Z))',
                    re.IGNORECASE | re.DOTALL
                )
                match = pattern.search(content)
                if match:
                    sections[section_name] = match.group().strip()
                    break
        
        # If no sections found, use heuristics
        if not sections:
            lines = content.split('\n')
            total_lines = len(lines)
            
            if total_lines > 100:
                # Assume first 40% is management remarks
                sections['management_remarks'] = '\n'.join(lines[:int(total_lines * 0.4)])
                # Last 50% is likely Q&A
                sections['qa'] = '\n'.join(lines[int(total_lines * 0.5):])
                # Middle might be transition/financial details
                sections['financial_results'] = '\n'.join(
                    lines[int(total_lines * 0.4):int(total_lines * 0.5)]
                )
            else:
                # Too short to meaningfully split
                sections['full_content'] = content
        
        return sections
    
    def process_document(self, file_path: str, content: str) -> Dict:
        """
        Process a single document into structured format
        
        Args:
            file_path: Path to the document
            content: Document text content
            
        Returns:
            Structured document dictionary
        """
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        # Extract sections
        sections = self.extract_sections(content)
        
        # Calculate metrics
        words = content.split()
        word_count = len(words)
        char_count = len(content)
        estimated_tokens = char_count // 4  # Rough estimate
        
        return {
            'metadata': metadata,
            'content': content,
            'sections': sections,
            'metrics': {
                'word_count': word_count,
                'char_count': char_count,
                'estimated_tokens': estimated_tokens,
                'section_count': len(sections)
            }
        }
    
    def process_multiple_documents(self, file_paths: List[str], 
                                 contents: Dict[str, str]) -> List[Dict]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of file paths
            contents: Dictionary mapping file_path to content
            
        Returns:
            List of structured documents
        """
        documents = []
        for file_path in file_paths:
            if file_path in contents:
                doc = self.process_document(file_path, contents[file_path])
                documents.append(doc)
        
        return documents
    
    def group_documents_by_bank(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Group documents by bank name"""
        grouped = {}
        for doc in documents:
            bank = doc['metadata']['bank_name']
            if bank not in grouped:
                grouped[bank] = []
            grouped[bank].append(doc)
        return grouped
    
    def group_documents_by_period(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Group documents by time period"""
        grouped = {}
        for doc in documents:
            period = doc['metadata']['period']
            if period not in grouped:
                grouped[period] = []
            grouped[period].append(doc)
        return grouped
    
    def sort_documents_by_date(self, documents: List[Dict]) -> List[Dict]:
        """Sort documents chronologically"""
        def get_sort_key(doc):
            period = doc['metadata']['period']
            # Extract year and quarter for sorting
            year = doc['metadata'].get('year', '0000')
            quarter = doc['metadata'].get('quarter', 'Q0')
            
            # Convert to sortable format (e.g., "2025Q1" -> 202501)
            quarter_num = quarter[1] if quarter and len(quarter) > 1 else '0'
            return f"{year}{quarter_num}"
        
        return sorted(documents, key=get_sort_key)