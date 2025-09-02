"""
Source Tracking and Citation System for RiskRadar
Tracks document sources, extracts quotes, and manages references
"""

import os
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

@dataclass
class DocumentReference:
    """Represents a reference to a source document"""
    file_path: str
    file_name: str
    document_type: str  # 'transcript', 'report', 'regulatory'
    company: str
    period: str  # e.g., 'Q3 2024'
    page_number: Optional[int] = None
    line_number: Optional[int] = None
    section: Optional[str] = None  # e.g., 'Q&A', 'Risk Management'
    
    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'document_type': self.document_type,
            'company': self.company,
            'period': self.period,
            'page_number': self.page_number,
            'line_number': self.line_number,
            'section': self.section
        }
    
    def get_citation(self) -> str:
        """Generate a formatted citation"""
        citation = f"{self.company} - {self.period}"
        if self.section:
            citation += f" ({self.section})"
        if self.page_number:
            citation += f", p.{self.page_number}"
        elif self.line_number:
            citation += f", line {self.line_number}"
        return citation

@dataclass
class SourceQuote:
    """Represents a quote extracted from a document"""
    text: str
    reference: DocumentReference
    context: str  # Surrounding text for context
    confidence: float  # Confidence in the extraction
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'reference': self.reference.to_dict(),
            'context': self.context,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class AnalysisSource:
    """Tracks sources for a complete analysis"""
    analysis_id: str
    primary_documents: List[DocumentReference]
    quotes: List[SourceQuote]
    metrics: Dict[str, List[DocumentReference]]  # metric_name -> sources
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'analysis_id': self.analysis_id,
            'primary_documents': [d.to_dict() for d in self.primary_documents],
            'quotes': [q.to_dict() for q in self.quotes],
            'metrics': {
                k: [ref.to_dict() for ref in v] 
                for k, v in self.metrics.items()
            },
            'timestamp': self.timestamp.isoformat()
        }

class SourceTracker:
    """Manages source tracking throughout the analysis pipeline"""
    
    def __init__(self):
        self.current_analysis: Optional[AnalysisSource] = None
        self.document_cache: Dict[str, DocumentReference] = {}
        self.quote_cache: List[SourceQuote] = []
        
    def start_analysis(self, analysis_id: str = None) -> str:
        """Start tracking a new analysis session"""
        if not analysis_id:
            analysis_id = self._generate_analysis_id()
        
        self.current_analysis = AnalysisSource(
            analysis_id=analysis_id,
            primary_documents=[],
            quotes=[],
            metrics={}
        )
        return analysis_id
    
    def add_document(self, file_path: str, document_type: str = 'transcript') -> DocumentReference:
        """Register a document being analyzed"""
        if file_path in self.document_cache:
            doc_ref = self.document_cache[file_path]
        else:
            # Extract metadata from filename
            file_name = os.path.basename(file_path)
            company, period = self._extract_metadata(file_name)
            
            doc_ref = DocumentReference(
                file_path=file_path,
                file_name=file_name,
                document_type=document_type,
                company=company,
                period=period
            )
            self.document_cache[file_path] = doc_ref
        
        if self.current_analysis and doc_ref not in self.current_analysis.primary_documents:
            self.current_analysis.primary_documents.append(doc_ref)
        
        return doc_ref
    
    def add_quote(self, text: str, document_path: str, 
                  context: str = "", page: int = None, 
                  line: int = None, section: str = None,
                  confidence: float = 1.0) -> SourceQuote:
        """Add a quote with its source reference"""
        # Get or create document reference
        if document_path in self.document_cache:
            doc_ref = self.document_cache[document_path]
        else:
            doc_ref = self.add_document(document_path)
        
        # Update reference with specific location
        ref_with_location = DocumentReference(
            file_path=doc_ref.file_path,
            file_name=doc_ref.file_name,
            document_type=doc_ref.document_type,
            company=doc_ref.company,
            period=doc_ref.period,
            page_number=page,
            line_number=line,
            section=section
        )
        
        quote = SourceQuote(
            text=text,
            reference=ref_with_location,
            context=context or text,
            confidence=confidence
        )
        
        self.quote_cache.append(quote)
        if self.current_analysis:
            self.current_analysis.quotes.append(quote)
        
        return quote
    
    def link_metric_to_source(self, metric_name: str, document_path: str,
                             page: int = None, line: int = None):
        """Link a specific metric to its source document"""
        if not self.current_analysis:
            return
        
        doc_ref = self.add_document(document_path)
        
        # Create reference with specific location
        ref_with_location = DocumentReference(
            file_path=doc_ref.file_path,
            file_name=doc_ref.file_name,
            document_type=doc_ref.document_type,
            company=doc_ref.company,
            period=doc_ref.period,
            page_number=page,
            line_number=line
        )
        
        if metric_name not in self.current_analysis.metrics:
            self.current_analysis.metrics[metric_name] = []
        
        self.current_analysis.metrics[metric_name].append(ref_with_location)
    
    def get_sources_for_metric(self, metric_name: str) -> List[DocumentReference]:
        """Get all source documents for a specific metric"""
        if not self.current_analysis:
            return []
        
        return self.current_analysis.metrics.get(metric_name, [])
    
    def get_quotes_for_document(self, document_path: str) -> List[SourceQuote]:
        """Get all quotes extracted from a specific document"""
        return [
            q for q in self.quote_cache 
            if q.reference.file_path == document_path
        ]
    
    def get_analysis_summary(self) -> Dict:
        """Get a summary of all sources for current analysis"""
        if not self.current_analysis:
            return {}
        
        return {
            'analysis_id': self.current_analysis.analysis_id,
            'documents_analyzed': len(self.current_analysis.primary_documents),
            'document_list': [
                {
                    'name': doc.file_name,
                    'company': doc.company,
                    'period': doc.period,
                    'type': doc.document_type
                }
                for doc in self.current_analysis.primary_documents
            ],
            'quotes_extracted': len(self.current_analysis.quotes),
            'metrics_with_sources': list(self.current_analysis.metrics.keys()),
            'timestamp': self.current_analysis.timestamp.isoformat()
        }
    
    def export_citations(self, format: str = 'json') -> str:
        """Export all citations for the current analysis"""
        if not self.current_analysis:
            return "{}"
        
        if format == 'json':
            return json.dumps(self.current_analysis.to_dict(), indent=2)
        elif format == 'markdown':
            return self._format_markdown_citations()
        elif format == 'html':
            return self._format_html_citations()
        else:
            return json.dumps(self.current_analysis.to_dict())
    
    def _extract_metadata(self, filename: str) -> Tuple[str, str]:
        """Extract company and period from filename"""
        # Common patterns in filenames
        company = "Unknown"
        period = "Unknown"
        
        # Try to extract company name
        companies = ['JPMorgan', 'HSBC', 'Citigroup', 'Barclays', 
                    'Lloyds', 'NatWest', 'Santander', 'StandardChartered']
        for c in companies:
            if c.lower() in filename.lower():
                company = c
                break
        
        # Try to extract period (Q1-Q4, H1-H2, Annual)
        import re
        
        # Quarter pattern
        quarter_match = re.search(r'Q[1-4]\s*20\d{2}', filename, re.IGNORECASE)
        if quarter_match:
            period = quarter_match.group(0)
        else:
            # Year pattern
            year_match = re.search(r'20\d{2}', filename)
            if year_match:
                period = year_match.group(0)
        
        return company, period
    
    def _generate_analysis_id(self) -> str:
        """Generate a unique analysis ID"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}_{os.getpid()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _format_markdown_citations(self) -> str:
        """Format citations as Markdown"""
        if not self.current_analysis:
            return ""
        
        md = f"# Analysis Sources\n"
        md += f"**Analysis ID:** {self.current_analysis.analysis_id}\n"
        md += f"**Timestamp:** {self.current_analysis.timestamp}\n\n"
        
        md += "## Documents Analyzed\n"
        for doc in self.current_analysis.primary_documents:
            md += f"- {doc.get_citation()}\n"
        
        md += "\n## Key Quotes\n"
        for quote in self.current_analysis.quotes[:10]:  # Top 10 quotes
            md += f"> \"{quote.text}\"\n"
            md += f"> — {quote.reference.get_citation()}\n\n"
        
        md += "\n## Metrics Sources\n"
        for metric, sources in self.current_analysis.metrics.items():
            md += f"**{metric}:**\n"
            for source in sources:
                md += f"- {source.get_citation()}\n"
            md += "\n"
        
        return md
    
    def _format_html_citations(self) -> str:
        """Format citations as HTML"""
        if not self.current_analysis:
            return ""
        
        html = f"""
        <div class="analysis-sources">
            <h2>Analysis Sources</h2>
            <p><strong>Analysis ID:</strong> {self.current_analysis.analysis_id}</p>
            <p><strong>Timestamp:</strong> {self.current_analysis.timestamp}</p>
            
            <h3>Documents Analyzed</h3>
            <ul>
        """
        
        for doc in self.current_analysis.primary_documents:
            html += f'<li><a href="file://{doc.file_path}">{doc.get_citation()}</a></li>'
        
        html += """
            </ul>
            
            <h3>Key Quotes</h3>
            <div class="quotes">
        """
        
        for quote in self.current_analysis.quotes[:10]:
            html += f"""
                <blockquote>
                    <p>"{quote.text}"</p>
                    <cite>— {quote.reference.get_citation()}</cite>
                </blockquote>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html

# Global instance for easy access
source_tracker = SourceTracker()