import boto3
from config.settings import settings
import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)

class AWSTextractExtractor:
    def __init__(self):
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError("AWS credentials not configured")
        
        self.client = boto3.client(
            'textract',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
    
    def analyze_document(self, document_content: bytes) -> Dict[str, Any]:
        """Analyze document using AWS Textract"""
        try:
            response = self.client.analyze_document(
                Document={'Bytes': document_content},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"AWS Textract analysis failed: {e}")
            return {"line_items": [], "totals": {}, "confidence": 0.0}
    
    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """Parse AWS Textract response"""
        line_items = []
        blocks = response.get('Blocks', [])
        
        # Extract tables for line items
        tables = [block for block in blocks if block['BlockType'] == 'TABLE']
        
        for table in tables:
            line_items.extend(self._extract_line_items_from_table(table, blocks))
        
        # Extract forms for additional data
        forms = [block for block in blocks if block['BlockType'] == 'KEY_VALUE_SET']
        totals = self._extract_totals_from_forms(forms, blocks)
        
        return {
            "line_items": line_items,
            "totals": totals,
            "confidence": 0.85
        }
    
    def _extract_line_items_from_table(self, table: Dict, blocks: List[Dict]) -> List[Dict]:
        """Extract line items from table structure"""
        line_items = []
        
        try:
            # Get all cells in this table
            table_cells = []
            for relationship in table.get('Relationships', []):
                if relationship['Type'] == 'CHILD':
                    for cell_id in relationship['Ids']:
                        cell_block = next((b for b in blocks if b['Id'] == cell_id), None)
                        if cell_block and cell_block['BlockType'] == 'CELL':
                            table_cells.append(cell_block)
            
            # Group cells by row
            rows = {}
            for cell in table_cells:
                row_index = cell.get('RowIndex', 0)
                if row_index not in rows:
                    rows[row_index] = []
                rows[row_index].append(cell)
            
            # Convert rows to line items (skip header row)
            for row_index, row_cells in sorted(rows.items()):
                if row_index > 1:  # Skip header
                    line_item = self._parse_table_row(row_cells, blocks)
                    if line_item:
                        line_items.append(line_item)
                        
        except Exception as e:
            logger.warning(f"Error extracting line items from table: {e}")
        
        return line_items
    
    def _parse_table_row(self, row_cells: List[Dict], blocks: List[Dict]) -> Dict:
        """Parse a table row into a line item"""
        try:
            description = ""
            quantity = 1.0
            rate = 0.0
            amount = 0.0
            
            for cell in row_cells:
                cell_text = self._get_cell_text(cell, blocks)
                if not cell_text:
                    continue
                    
                # Simple heuristic-based parsing
                if not description and len(cell_text) > 2 and not self._looks_like_number(cell_text):
                    description = cell_text
                elif self._looks_like_amount(cell_text):
                    parsed_amount = self._parse_amount(cell_text)
                    if amount == 0.0:
                        amount = parsed_amount
                    else:
                        rate = parsed_amount
            
            if amount > 0 and description:
                return {
                    "item_name": description,
                    "item_quantity": quantity,
                    "item_rate": rate if rate > 0 else amount,
                    "item_amount": amount,
                    "confidence": 0.8
                }
        except Exception as e:
            logger.warning(f"Failed to parse table row: {e}")
        
        return None
    
    def _get_cell_text(self, cell: Dict, blocks: List[Dict]) -> str:
        """Extract text from cell"""
        text = ""
        relationships = cell.get('Relationships', [])
        for relationship in relationships:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    child_block = next((b for b in blocks if b['Id'] == child_id), None)
                    if child_block and child_block['BlockType'] == 'WORD':
                        text += child_block.get('Text', '') + " "
        return text.strip()
    
    def _looks_like_number(self, text: str) -> bool:
        """Check if text looks like a number"""
        return bool(re.match(r'^[\d.,]+$', text.strip()))
    
    def _looks_like_amount(self, text: str) -> bool:
        """Check if text looks like a monetary amount"""
        return bool(re.match(r'^[\$€£]?[\d,]+\.?\d*$', text.strip()))
    
    def _parse_amount(self, text: str) -> float:
        """Parse amount string to float"""
        try:
            cleaned = re.sub(r'[^\d.]', '', text)
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    
    def _extract_totals_from_forms(self, forms: List[Dict], blocks: List[Dict]) -> Dict:
        """Extract totals from form fields"""
        totals = {}
        
        try:
            for form in forms:
                if form.get('EntityTypes', []) and 'KEY' in form.get('EntityTypes', []):
                    key_text = self._get_cell_text(form, blocks).lower()
                    if any(total_keyword in key_text for total_keyword in ['total', 'amount', 'balance']):
                        # Find corresponding value
                        for relationship in form.get('Relationships', []):
                            if relationship['Type'] == 'VALUE':
                                for value_id in relationship['Ids']:
                                    value_block = next((b for b in blocks if b['Id'] == value_id), None)
                                    if value_block:
                                        value_text = self._get_cell_text(value_block, blocks)
                                        amount = self._parse_amount(value_text)
                                        if amount > 0:
                                            totals['Total'] = amount
        except Exception as e:
            logger.warning(f"Error extracting totals from forms: {e}")
        
        return totals