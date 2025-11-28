from typing import Dict, Any, List
import logging
from src.extraction.mock_extractor import MockExtractor
from src.reconciliation.validator import ReconciliationEngine

logger = logging.getLogger(__name__)

class BillExtractionPipeline:
    def __init__(self, use_mock: bool = True):
        self.reconciliation_engine = ReconciliationEngine()
        self.extractor = MockExtractor()
        logger.info("Using mock extractor for bill extraction")
        
    def process_document(self, document_url: str) -> Dict[str, Any]:
        """Process document and extract bill data"""
        try:
            logger.info(f"Processing document: {document_url}")
            
            # Use mock extractor directly
            extraction_result = self.extractor.analyze_document(b"mock")
            
            # Reconcile and validate
            reconciliation_result = self.reconciliation_engine.reconcile_extraction(
                extraction_result, 
                extraction_result.get('totals', {})
            )
            
            return self._format_success_response(reconciliation_result)
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return self._error_response(f"Processing error: {str(e)}")
    
    def _format_success_response(self, reconciliation_result: Dict) -> Dict[str, Any]:
        """Format successful response"""
        line_items = reconciliation_result['line_items']
        
        formatted_items = []
        for item in line_items:
            formatted_items.append({
                "item_name": item["item_name"],
                "item_amount": round(float(item["item_amount"]), 2),
                "item_rate": round(float(item.get("item_rate", item["item_amount"])), 2),
                "item_quantity": float(item.get("item_quantity", 1.0))
            })
        
        pagewise_line_items = [{
            "page_no": "1",
            "bill_items": formatted_items
        }]
        
        return {
            "is_success": True,
            "data": {
                "pagewise_line_items": pagewise_line_items,
                "total_item_count": len(formatted_items),
                "reconciled_amount": round(float(reconciliation_result['reconciled_amount']), 2)
            }
        }
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "is_success": False,
            "error": error_message,
            "data": None
        }
