from typing import Dict, Any, List
import logging
from src.preprocessing.document_processor import DocumentProcessor
from src.extraction.azure_extractor import AzureFormRecognizerExtractor
from src.extraction.aws_extractor import AWSTextractExtractor
from src.extraction.mock_extractor import MockExtractor
from src.reconciliation.validator import ReconciliationEngine
from config.settings import settings

logger = logging.getLogger(__name__)

class BillExtractionPipeline:
    def __init__(self, use_mock: bool = False):
        self.document_processor = DocumentProcessor()
        self.reconciliation_engine = ReconciliationEngine()
        
        # Initialize extractors based on configuration
        self.use_mock = use_mock or settings.use_mock
        
        if self.use_mock:
            logger.info("Using mock extractor for testing")
            self.primary_extractor = MockExtractor()
            self.fallback_extractor = MockExtractor()
        else:
            try:
                self.primary_extractor = AzureFormRecognizerExtractor()
                self.fallback_extractor = AWSTextractExtractor()
                logger.info("Initialized Azure Form Recognizer and AWS Textract")
            except Exception as e:
                logger.warning(f"Failed to initialize cloud extractors: {e}")
                logger.info("Falling back to mock extractor")
                self.primary_extractor = MockExtractor()
                self.fallback_extractor = MockExtractor()
                self.use_mock = True
        
    def process_document(self, document_url: str) -> Dict[str, Any]:
        """Main pipeline to process document and extract bill data"""
        try:
            logger.info(f"Starting pipeline processing for: {document_url}")
            
            # Step 1: Download document
            document_content = self.document_processor.download_document(document_url)
            if not document_content:
                return self._error_response("Failed to download document from URL")
            
            # Step 2: Validate document
            if not self.document_processor.validate_document(document_content):
                return self._error_response("Invalid document format or size")
            
            # Step 3: Preprocess document
            processed_content, format_info = self.document_processor.preprocess_image(document_content)
            logger.info(f"Document processed successfully. Format: {format_info}")
            
            # Step 4: Extract with primary model
            logger.info("Starting primary extraction...")
            primary_result = self.primary_extractor.analyze_document(processed_content)
            
            # Step 5: If low confidence or no items, use fallback
            if (primary_result['confidence'] < settings.CONFIDENCE_THRESHOLD or 
                len(primary_result['line_items']) == 0):
                
                logger.info(f"Primary extraction low confidence ({primary_result['confidence']}), using fallback...")
                fallback_result = self.fallback_extractor.analyze_document(processed_content)
                
                # Choose the better result
                if (fallback_result['confidence'] > primary_result['confidence'] and 
                    len(fallback_result['line_items']) > 0):
                    extraction_result = fallback_result
                    logger.info(f"Using fallback result with confidence: {fallback_result['confidence']}")
                else:
                    extraction_result = primary_result
                    logger.info(f"Sticking with primary result with confidence: {primary_result['confidence']}")
            else:
                extraction_result = primary_result
                logger.info(f"Primary extraction successful with confidence: {primary_result['confidence']}")
            
            # Step 6: Reconcile and validate
            logger.info("Starting reconciliation...")
            reconciliation_result = self.reconciliation_engine.reconcile_extraction(
                extraction_result, 
                extraction_result.get('totals', {})
            )
            
            # Step 7: Format response
            response = self._format_success_response(reconciliation_result)
            logger.info(f"Pipeline completed successfully. Items extracted: {len(reconciliation_result['line_items'])}")
            
            return response
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}", exc_info=True)
            return self._error_response(f"Processing error: {str(e)}")
    
    def _format_success_response(self, reconciliation_result: Dict) -> Dict[str, Any]:
        """Format successful response according to API specification"""
        line_items = reconciliation_result['line_items']
        
        # Format line items to match expected schema
        formatted_items = []
        for item in line_items:
            formatted_items.append({
                "item_name": item["item_name"],
                "item_amount": round(float(item["item_amount"]), 2),
                "item_rate": round(float(item.get("item_rate", item["item_amount"])), 2),
                "item_quantity": float(item.get("item_quantity", 1.0))
            })
        
        # Group by page (for now, assume single page)
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
        """Format error response"""
        return {
            "is_success": False,
            "error": error_message,
            "data": None
        }