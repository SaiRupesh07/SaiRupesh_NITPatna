import logging
from typing import List, Dict, Any
import random

logger = logging.getLogger(__name__)

class MockExtractor:
    """Mock extractor for testing without real API keys"""
    
    def __init__(self):
        self.sample_items = [
            {"item_name": "Livi 300ng Tab", "item_rate": 22.0, "item_quantity": 14, "item_amount": 308.0},
            {"item_name": "Meinuro", "item_rate": 17.72, "item_quantity": 7, "item_amount": 124.84},
            {"item_name": "Pizat 4.5", "item_rate": 419.86, "item_quantity": 2, "item_amount": 839.72},
            {"item_name": "Supralite Q8 Syr", "item_rate": 289.69, "item_quantity": 1, "item_amount": 289.69},
            {"item_name": "Consultation Fee", "item_rate": 150.0, "item_quantity": 1, "item_amount": 150.0},
            {"item_name": "Lab Test Basic", "item_rate": 200.0, "item_quantity": 1, "item_amount": 200.0}
        ]
    
    def analyze_document(self, document_content: bytes) -> Dict[str, Any]:
        """Mock document analysis"""
        try:
            # Simulate processing time
            import time
            time.sleep(1)
            
            # Randomly select 3-6 items
            num_items = random.randint(3, 6)
            selected_items = random.sample(self.sample_items, num_items)
            
            # Add some variation to amounts
            line_items = []
            for item in selected_items:
                varied_item = item.copy()
                # Add small random variation
                variation = random.uniform(0.95, 1.05)
                varied_item["item_amount"] = round(varied_item["item_rate"] * varied_item["item_quantity"] * variation, 2)
                varied_item["confidence"] = round(random.uniform(0.85, 0.98), 2)
                line_items.append(varied_item)
            
            # Calculate totals
            total_amount = sum(item["item_amount"] for item in line_items)
            
            return {
                "line_items": line_items,
                "totals": {
                    "Total": total_amount,
                    "Subtotal": round(total_amount * 0.9, 2),
                    "Tax": round(total_amount * 0.1, 2)
                },
                "confidence": round(random.uniform(0.88, 0.96), 2)
            }
            
        except Exception as e:
            logger.error(f"Mock extraction failed: {e}")
            return {"line_items": [], "totals": {}, "confidence": 0.0}