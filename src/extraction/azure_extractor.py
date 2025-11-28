from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from config.settings import settings
import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)

class AzureFormRecognizerExtractor:
    def __init__(self):
        if not settings.AZURE_FORM_RECOGNIZER_ENDPOINT or not settings.AZURE_FORM_RECOGNIZER_KEY:
            raise ValueError("Azure Form Recognizer credentials not configured")
        
        self.client = DocumentAnalysisClient(
            endpoint=settings.AZURE_FORM_RECOGNIZER_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_FORM_RECOGNIZER_KEY)
        )
    
    def analyze_document(self, document_content: bytes) -> Dict[str, Any]:
        """Analyze document using Azure Form Recognizer"""
        try:
            poller = self.client.begin_analyze_document(
                "prebuilt-invoice", 
                document_content
            )
            result = poller.result()
            
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Azure Form Recognizer analysis failed: {e}")
            return {"line_items": [], "totals": {}, "confidence": 0.0}
    
    def _parse_result(self, result) -> Dict[str, Any]:
        """Parse Azure Form Recognizer result"""
        line_items = []
        totals = {}
        confidence_scores = []
        
        # Extract line items from invoices
        if hasattr(result, 'documents') and result.documents:
            for doc in result.documents:
                line_items.extend(self._extract_line_items_from_document(doc))
                totals.update(self._extract_totals_from_document(doc))
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.8
        
        return {
            "line_items": line_items,
            "totals": totals,
            "confidence": avg_confidence
        }
    
    def _extract_line_items_from_document(self, document) -> List[Dict]:
        """Extract line items from document"""
        line_items = []
        
        try:
            # Look for items table in invoice
            if hasattr(document, 'fields'):
                items_field = document.fields.get('Items')
                if items_field and hasattr(items_field, 'value'):
                    items = items_field.value
                    if isinstance(items, list):
                        for item in items:
                            line_item = self._parse_invoice_item(item)
                            if line_item:
                                line_items.append(line_item)
        except Exception as e:
            logger.warning(f"Error extracting line items: {e}")
        
        return line_items
    
    def _parse_invoice_item(self, item) -> Dict:
        """Parse individual invoice line item"""
        try:
            description = ""
            quantity = 1.0
            unit_price = 0.0
            amount = 0.0
            
            if hasattr(item, 'value'):
                item_value = item.value
                
                # Extract description
                desc_field = item_value.get('Description')
                if desc_field and hasattr(desc_field, 'value'):
                    description = desc_field.value
                
                # Extract quantity
                qty_field = item_value.get('Quantity')
                if qty_field and hasattr(qty_field, 'value'):
                    try:
                        quantity = float(qty_field.value)
                    except (ValueError, TypeError):
                        quantity = 1.0
                
                # Extract unit price
                price_field = item_value.get('UnitPrice')
                if price_field and hasattr(price_field, 'value'):
                    try:
                        unit_price = float(price_field.value)
                    except (ValueError, TypeError):
                        unit_price = 0.0
                
                # Extract amount
                amount_field = item_value.get('Amount')
                if amount_field and hasattr(amount_field, 'value'):
                    try:
                        amount = float(amount_field.value)
                    except (ValueError, TypeError):
                        amount = 0.0
            
            # If amount is 0 but we have unit price and quantity, calculate it
            if amount == 0 and unit_price > 0:
                amount = unit_price * quantity
            
            if description and amount > 0:
                return {
                    "item_name": str(description),
                    "item_quantity": quantity,
                    "item_rate": unit_price,
                    "item_amount": amount,
                    "confidence": 0.9
                }
                
        except Exception as e:
            logger.warning(f"Failed to parse invoice item: {e}")
        
        return None
    
    def _extract_totals_from_document(self, document) -> Dict:
        """Extract totals from document"""
        totals = {}
        
        try:
            if hasattr(document, 'fields'):
                total_field = document.fields.get('Total')
                if total_field and hasattr(total_field, 'value'):
                    totals['Total'] = float(total_field.value)
                
                subtotal_field = document.fields.get('Subtotal')
                if subtotal_field and hasattr(subtotal_field, 'value'):
                    totals['Subtotal'] = float(subtotal_field.value)
                
                tax_field = document.fields.get('TotalTax')
                if tax_field and hasattr(tax_field, 'value'):
                    totals['Tax'] = float(tax_field.value)
                    
        except Exception as e:
            logger.warning(f"Error extracting totals: {e}")
        
        return totals