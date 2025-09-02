"""
Document Validator for RiskRadar
Validates document sets for different analysis modes
"""

from typing import Dict, List, Set, Tuple, Optional

class DocumentValidator:
    """Validate document sets for different analysis modes"""
    
    def __init__(self):
        self.valid_modes = [
            "Single Document Analysis",
            "Cross-Bank Comparison", 
            "Timeline Analysis"
        ]
    
    def validate_document_set(self, documents: List[Dict], mode: str) -> Dict:
        """
        Validate a set of documents for a specific analysis mode
        
        Args:
            documents: List of structured documents from DocumentProcessor
            mode: Analysis mode to validate for
            
        Returns:
            Validation result with 'valid' flag and 'message'
        """
        if not documents:
            return {
                'valid': False,
                'message': "No documents selected. Please select at least one document."
            }
        
        if mode not in self.valid_modes:
            return {
                'valid': False,
                'message': f"Invalid analysis mode: {mode}"
            }
        
        if mode == "Single Document Analysis":
            return self._validate_single_document(documents)
        elif mode == "Cross-Bank Comparison":
            return self._validate_cross_bank(documents)
        elif mode == "Timeline Analysis":
            return self._validate_timeline(documents)
    
    def _validate_single_document(self, documents: List[Dict]) -> Dict:
        """Validate for single document analysis"""
        if len(documents) != 1:
            return {
                'valid': False,
                'message': f"Single Document Analysis requires exactly 1 document. You selected {len(documents)}.",
                'suggestion': "Please select only one document for this analysis mode."
            }
        
        return {'valid': True, 'message': "Document validated successfully"}
    
    def _validate_cross_bank(self, documents: List[Dict]) -> Dict:
        """Validate for cross-bank comparison"""
        if len(documents) < 2:
            return {
                'valid': False,
                'message': "Cross-Bank Comparison requires at least 2 documents.",
                'suggestion': "Please select documents from multiple banks to compare."
            }
        
        # Check that all documents are from the same period
        periods = {doc['metadata']['period'] for doc in documents}
        if len(periods) > 1:
            period_list = sorted(list(periods))
            return {
                'valid': False,
                'message': f"Cross-Bank Comparison requires all documents from the same period.",
                'details': f"Found documents from: {', '.join(period_list)}",
                'suggestion': "Please select documents from the same quarter/year for meaningful comparison."
            }
        
        # Check that documents are from different banks
        banks = {doc['metadata']['bank_name'] for doc in documents}
        if len(banks) < 2:
            bank_list = sorted(list(banks))
            return {
                'valid': False,
                'message': "Cross-Bank Comparison requires documents from different banks.",
                'details': f"All selected documents are from: {', '.join(bank_list)}",
                'suggestion': "Please select documents from at least 2 different banks."
            }
        
        # Success - valid cross-bank comparison
        period = list(periods)[0]
        bank_list = sorted(list(banks))
        return {
            'valid': True,
            'message': f"Valid cross-bank comparison for {period}",
            'details': f"Comparing {len(banks)} banks: {', '.join(bank_list)}"
        }
    
    def _validate_timeline(self, documents: List[Dict]) -> Dict:
        """Validate for timeline analysis"""
        if len(documents) < 2:
            return {
                'valid': False,
                'message': "Timeline Analysis requires at least 2 documents.",
                'suggestion': "Please select multiple documents from different time periods."
            }
        
        # Check that all documents are from the same bank
        banks = {doc['metadata']['bank_name'] for doc in documents}
        if len(banks) > 1:
            bank_list = sorted(list(banks))
            return {
                'valid': False,
                'message': "Timeline Analysis requires all documents from the same bank.",
                'details': f"Found documents from: {', '.join(bank_list)}",
                'suggestion': "Please select documents from only one bank to track evolution over time."
            }
        
        # Check that documents are from different periods
        periods = {doc['metadata']['period'] for doc in documents}
        if len(periods) < 2:
            period = list(periods)[0] if periods else "Unknown"
            return {
                'valid': False,
                'message': "Timeline Analysis requires documents from different time periods.",
                'details': f"All selected documents are from: {period}",
                'suggestion': "Please select documents from at least 2 different quarters/years."
            }
        
        # Success - valid timeline analysis
        bank = list(banks)[0]
        period_list = sorted(list(periods))
        return {
            'valid': True,
            'message': f"Valid timeline analysis for {bank}",
            'details': f"Analyzing {len(periods)} periods: {', '.join(period_list)}"
        }
    
    def suggest_mode(self, documents: List[Dict]) -> str:
        """
        Suggest the best analysis mode based on selected documents
        
        Args:
            documents: List of structured documents
            
        Returns:
            Suggested analysis mode
        """
        if not documents:
            return "Single Document Analysis"
        
        if len(documents) == 1:
            return "Single Document Analysis"
        
        # Check if same bank (timeline potential)
        banks = {doc['metadata']['bank_name'] for doc in documents}
        periods = {doc['metadata']['period'] for doc in documents}
        
        if len(banks) == 1 and len(periods) > 1:
            return "Timeline Analysis"
        
        if len(banks) > 1 and len(periods) == 1:
            return "Cross-Bank Comparison"
        
        # Mixed case - default to single document
        return "Single Document Analysis"
    
    def get_mode_requirements(self, mode: str) -> Dict:
        """
        Get requirements for a specific analysis mode
        
        Args:
            mode: Analysis mode
            
        Returns:
            Dictionary describing mode requirements
        """
        requirements = {
            "Single Document Analysis": {
                'min_documents': 1,
                'max_documents': 1,
                'same_bank': False,
                'same_period': False,
                'description': "Analyze a single earnings call or report in detail",
                'use_case': "Deep dive into one specific document"
            },
            "Cross-Bank Comparison": {
                'min_documents': 2,
                'max_documents': None,
                'same_bank': False,
                'same_period': True,
                'different_banks': True,
                'description': "Compare multiple banks from the same time period",
                'use_case': "Identify systemic risks and sector-wide patterns"
            },
            "Timeline Analysis": {
                'min_documents': 2,
                'max_documents': None,
                'same_bank': True,
                'same_period': False,
                'different_periods': True,
                'description': "Track one bank's evolution over multiple periods",
                'use_case': "Detect risk trends and early warning signals"
            }
        }
        
        return requirements.get(mode, {})
    
    def filter_documents_for_mode(self, documents: List[Dict], mode: str,
                                 selected_bank: Optional[str] = None,
                                 selected_period: Optional[str] = None) -> List[Dict]:
        """
        Filter available documents based on mode requirements
        
        Args:
            documents: List of all available documents
            mode: Analysis mode
            selected_bank: For timeline mode, filter by this bank
            selected_period: For cross-bank mode, filter by this period
            
        Returns:
            Filtered list of documents suitable for the mode
        """
        if mode == "Single Document Analysis":
            # All documents are valid for single analysis
            return documents
        
        elif mode == "Cross-Bank Comparison":
            # Filter by selected period if provided
            if selected_period:
                return [doc for doc in documents 
                       if doc['metadata']['period'] == selected_period]
            return documents
        
        elif mode == "Timeline Analysis":
            # Filter by selected bank if provided
            if selected_bank:
                return [doc for doc in documents 
                       if doc['metadata']['bank_name'] == selected_bank]
            return documents
        
        return documents
    
    def get_validation_summary(self, documents: List[Dict]) -> Dict:
        """
        Get a summary of what analysis modes are possible with selected documents
        
        Args:
            documents: List of structured documents
            
        Returns:
            Summary of valid modes and reasons
        """
        summary = {
            'selected_count': len(documents),
            'modes': {}
        }
        
        # Check each mode
        for mode in self.valid_modes:
            validation = self.validate_document_set(documents, mode)
            summary['modes'][mode] = {
                'valid': validation['valid'],
                'message': validation.get('message', ''),
                'details': validation.get('details', ''),
                'suggestion': validation.get('suggestion', '')
            }
        
        # Add document statistics
        if documents:
            banks = {doc['metadata']['bank_name'] for doc in documents}
            periods = {doc['metadata']['period'] for doc in documents}
            
            summary['statistics'] = {
                'unique_banks': len(banks),
                'banks': sorted(list(banks)),
                'unique_periods': len(periods),
                'periods': sorted(list(periods))
            }
        
        return summary