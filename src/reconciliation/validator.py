import logging
from typing import List, Dict, Any
from difflib import SequenceMatcher
import numpy as np

logger = logging.getLogger(__name__)

class ReconciliationEngine:
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
    
    def reconcile_extraction(self, extracted_data: Dict, document_totals: Dict) -> Dict[str, Any]:
        """Reconcile extracted data with document totals"""
        line_items = extracted_data.get("line_items", [])
        
        # Remove duplicates
        unique_items = self._remove_duplicates(line_items)
        
        # Calculate totals
        calculated_total = sum(item['item_amount'] for item in unique_items)
        extracted_total = self._get_extracted_total(document_totals)
        
        # Check reconciliation
        discrepancy = abs(calculated_total - extracted_total)
        is_reconciled = discrepancy <= (extracted_total * self.tolerance) if extracted_total > 0 else True
        
        reconciled_amount = extracted_total if is_reconciled else calculated_total
        
        logger.info(f"Reconciliation: Calculated=${calculated_total:.2f}, "
                   f"Extracted=${extracted_total:.2f}, Discrepancy=${discrepancy:.2f}, "
                   f"Reconciled=${reconciled_amount:.2f}")
        
        return {
            "line_items": unique_items,
            "calculated_total": calculated_total,
            "extracted_total": extracted_total,
            "discrepancy": discrepancy,
            "is_reconciled": is_reconciled,
            "reconciled_amount": reconciled_amount
        }
    
    def _remove_duplicates(self, line_items: List[Dict]) -> List[Dict]:
        """Remove duplicate line items using fuzzy matching"""
        unique_items = []
        
        for item in line_items:
            if not self._is_duplicate(item, unique_items):
                unique_items.append(item)
        
        logger.info(f"Removed {len(line_items) - len(unique_items)} duplicates")
        return unique_items
    
    def _is_duplicate(self, item: Dict, existing_items: List[Dict]) -> bool:
        """Check if item is duplicate using fuzzy matching"""
        for existing in existing_items:
            # Check name similarity
            name_similarity = SequenceMatcher(
                None, 
                item['item_name'].lower(), 
                existing['item_name'].lower()
            ).ratio()
            
            # Check amount similarity (handle division by zero)
            max_amount = max(abs(item['item_amount']), abs(existing['item_amount']))
            if max_amount > 0:
                amount_similarity = 1 - abs(item['item_amount'] - existing['item_amount']) / max_amount
            else:
                amount_similarity = 1.0
            
            # Consider duplicate if both name and amount are very similar
            if name_similarity > 0.9 and amount_similarity > 0.95:
                logger.debug(f"Found duplicate: {item['item_name']} (similarity: {name_similarity:.2f})")
                return True
        
        return False
    
    def _get_extracted_total(self, document_totals: Dict) -> float:
        """Extract the final total from document totals"""
        total_fields = ['Total', 'AmountDue', 'InvoiceTotal', 'FinalTotal', 'GrandTotal']
        
        for field in total_fields:
            if field in document_totals:
                total = document_totals[field]
                logger.info(f"Found total in field '{field}': ${total:.2f}")
                return total
        
        # If no total found, return 0 (will use calculated total)
        logger.warning("No total found in document totals, using calculated total")
        return 0.0