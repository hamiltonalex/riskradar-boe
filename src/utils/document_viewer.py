"""
Document Viewer and Preview Component for RiskRadar
Handles document display, quote highlighting, and navigation
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import PyPDF2
import pdfplumber

class DocumentViewer:
    """Handles document viewing and content extraction"""
    
    def __init__(self):
        self.cached_documents: Dict[str, str] = {}
        self.document_metadata: Dict[str, Dict] = {}
    
    def load_document(self, file_path: str, cache: bool = True) -> Dict:
        """Load a document and extract its content"""
        if cache and file_path in self.cached_documents:
            return {
                'content': self.cached_documents[file_path],
                'metadata': self.document_metadata.get(file_path, {})
            }
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            content, metadata = self._load_pdf(file_path)
        elif file_extension == '.txt':
            content, metadata = self._load_text(file_path)
        else:
            content = f"Unsupported file type: {file_extension}"
            metadata = {}
        
        if cache:
            self.cached_documents[file_path] = content
            self.document_metadata[file_path] = metadata
        
        return {'content': content, 'metadata': metadata}
    
    def _load_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """Load PDF and extract text with page information"""
        content = ""
        metadata = {
            'type': 'pdf',
            'pages': [],
            'total_pages': 0
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata['total_pages'] = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Add page marker for reference
                        content += f"\n[PAGE {i}]\n{page_text}\n"
                        metadata['pages'].append({
                            'number': i,
                            'start_pos': len(content) - len(page_text),
                            'end_pos': len(content)
                        })
        except:
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata['total_pages'] = len(pdf_reader.pages)
                    
                    for i, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        content += f"\n[PAGE {i}]\n{page_text}\n"
                        metadata['pages'].append({
                            'number': i,
                            'start_pos': len(content) - len(page_text),
                            'end_pos': len(content)
                        })
            except Exception as e:
                content = f"Error loading PDF: {str(e)}"
        
        return content, metadata
    
    def _load_text(self, file_path: str) -> Tuple[str, Dict]:
        """Load text file with line information"""
        metadata = {
            'type': 'text',
            'lines': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                metadata['lines'] = len(content.split('\n'))
        except Exception as e:
            content = f"Error loading text file: {str(e)}"
        
        return content, metadata
    
    def extract_context(self, file_path: str, search_text: str, 
                       context_lines: int = 3) -> List[Dict]:
        """Extract text with surrounding context"""
        document = self.load_document(file_path)
        content = document['content']
        
        if not content:
            return []
        
        # Split content into lines
        lines = content.split('\n')
        results = []
        
        # Search for the text
        search_pattern = re.compile(re.escape(search_text), re.IGNORECASE)
        
        for i, line in enumerate(lines):
            if search_pattern.search(line):
                # Extract context
                start_idx = max(0, i - context_lines)
                end_idx = min(len(lines), i + context_lines + 1)
                
                context = {
                    'line_number': i + 1,
                    'matched_line': line,
                    'context': '\n'.join(lines[start_idx:end_idx]),
                    'page': self._get_page_number(document, i)
                }
                results.append(context)
        
        return results
    
    def _get_page_number(self, document: Dict, line_index: int) -> Optional[int]:
        """Get page number for a given line index"""
        metadata = document.get('metadata', {})
        
        if metadata.get('type') == 'pdf':
            # Find which page contains this line
            content = document['content']
            lines = content.split('\n')
            
            # Look for page markers
            for i in range(line_index, -1, -1):
                if lines[i].startswith('[PAGE '):
                    page_match = re.match(r'\[PAGE (\d+)\]', lines[i])
                    if page_match:
                        return int(page_match.group(1))
        
        return None
    
    def highlight_text(self, content: str, highlights: List[str]) -> str:
        """Add HTML highlighting to specific text segments"""
        highlighted_content = content
        
        for highlight in highlights:
            # Escape special regex characters
            pattern = re.escape(highlight)
            # Add HTML highlighting
            replacement = f'<mark style="background-color: yellow;">{highlight}</mark>'
            highlighted_content = re.sub(
                pattern, 
                replacement, 
                highlighted_content, 
                flags=re.IGNORECASE
            )
        
        return highlighted_content
    
    def get_preview(self, file_path: str, max_chars: int = 1000) -> str:
        """Get a preview of the document"""
        document = self.load_document(file_path)
        content = document['content']
        
        if len(content) <= max_chars:
            return content
        
        # Try to find a good break point
        preview = content[:max_chars]
        last_period = preview.rfind('.')
        if last_period > max_chars * 0.8:  # If period is near the end
            preview = preview[:last_period + 1]
        
        return preview + "..."
    
    def search_in_document(self, file_path: str, search_terms: List[str]) -> Dict:
        """Search for multiple terms in a document"""
        document = self.load_document(file_path)
        content = document['content'].lower()
        
        results = {
            'file': file_path,
            'matches': {}
        }
        
        for term in search_terms:
            term_lower = term.lower()
            count = content.count(term_lower)
            
            if count > 0:
                # Find positions
                positions = []
                start = 0
                while True:
                    pos = content.find(term_lower, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                results['matches'][term] = {
                    'count': count,
                    'positions': positions[:10]  # First 10 occurrences
                }
        
        return results
    
    def create_document_link(self, file_path: str, page: int = None, 
                            line: int = None) -> str:
        """Create a navigable link to a document location"""
        # Clean the path
        abs_path = os.path.abspath(file_path)
        
        # Create link based on file type
        if file_path.endswith('.pdf') and page:
            # PDF with page number
            link = f"file://{abs_path}#page={page}"
        else:
            # Generic file link
            link = f"file://{abs_path}"
        
        # Add line reference if available
        if line:
            link += f"#line={line}"
        
        return link
    
    def format_source_card(self, file_path: str, quote: str = None,
                          page: int = None, line: int = None) -> str:
        """Format a source reference as an HTML card"""
        file_name = os.path.basename(file_path)
        link = self.create_document_link(file_path, page, line)
        
        card_html = f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin-top: 0;">📄 {file_name}</h4>
        """
        
        if quote:
            card_html += f"""
            <blockquote style="border-left: 3px solid #4CAF50; padding-left: 10px; margin: 10px 0;">
                "{quote}"
            </blockquote>
            """
        
        if page:
            card_html += f"<p>📍 Page {page}"
            if line:
                card_html += f", Line {line}"
            card_html += "</p>"
        elif line:
            card_html += f"<p>📍 Line {line}</p>"
        
        card_html += f"""
            <a href="{link}" target="_blank" style="color: #1976D2; text-decoration: none;">
                🔗 Open Document
            </a>
        </div>
        """
        
        return card_html

# Global instance
document_viewer = DocumentViewer()