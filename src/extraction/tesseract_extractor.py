import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import requests
import io
import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional

class TesseractExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.medical_terms = ['tab', 'cap', 'syr', 'inj', 'mg', 'ml', 'medicine', 'drug', 'pharma', 'tablet', 'capsule', 'syrup', 'injection']
        
    def extract_text_from_content(self, document_content: bytes) -> str:
        """Enhanced OCR with better preprocessing"""
        try:
            image = Image.open(io.BytesIO(document_content))
            processed_image = self._advanced_preprocessing(image)
            
            # Multiple OCR attempts with different configurations
            text = self._robust_ocr(processed_image)
            
            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _advanced_preprocessing(self, image: Image.Image) -> Image.Image:
        """Advanced image preprocessing for better OCR"""
        try:
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize for better resolution
            if image.size[0] < 1000:
                scale_factor = 2000 / image.size[0]
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(3))
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Advanced preprocessing failed: {e}")
            return image
    
    def _robust_ocr(self, image: Image.Image) -> str:
        """Multiple OCR attempts with different configurations"""
        ocr_results = []
        
        # Configuration 1: Default for invoices
        try:
            config1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,$ ()/-+'
            text1 = pytesseract.image_to_string(image, config=config1)
            if self._is_quality_text(text1):
                ocr_results.append(("config1", text1))
        except Exception as e:
            self.logger.debug(f"OCR config1 failed: {e}")
        
        # Configuration 2: Single text line mode
        try:
            config2 = r'--oem 3 --psm 8'
            text2 = pytesseract.image_to_string(image, config=config2)
            if self._is_quality_text(text2):
                ocr_results.append(("config2", text2))
        except Exception as e:
            self.logger.debug(f"OCR config2 failed: {e}")
        
        # Configuration 3: Sparse text
        try:
            config3 = r'--oem 3 --psm 11'
            text3 = pytesseract.image_to_string(image, config=config3)
            if self._is_quality_text(text3):
                ocr_results.append(("config3", text3))
        except Exception as e:
            self.logger.debug(f"OCR config3 failed: {e}")
        
        # Choose the best result
        if ocr_results:
            # Prioritize results with numbers and medical terms
            scored_results = []
            for config_name, text in ocr_results:
                score = self._score_text_quality(text)
                scored_results.append((score, text))
            
            # Return highest scored text
            scored_results.sort(reverse=True)
            return scored_results[0][1]
        
        return ""
    
    def _is_quality_text(self, text: str) -> bool:
        """Check if text contains meaningful content"""
        if not text or len(text.strip()) < 10:
            return False
        
        # Check for presence of numbers (likely prices/quantities)
        has_numbers = bool(re.search(r'\d+\.?\d*', text))
        
        # Check for reasonable word lengths
        words = text.split()
        if len(words) < 2:
            return False
            
        return has_numbers
    
    def _score_text_quality(self, text: str) -> float:
        """Score text quality based on medical invoice patterns"""
        score = 0.0
        
        # Bonus for numbers (prices, quantities)
        numbers = re.findall(r'\d+\.?\d*', text)
        score += len(numbers) * 0.2
        
        # Bonus for medical terms
        text_lower = text.lower()
        medical_matches = sum(1 for term in self.medical_terms if term in text_lower)
        score += medical_matches * 0.3
        
        # Bonus for currency symbols or common invoice terms
        invoice_terms = ['total', 'amount', 'rate', 'qty', 'quantity', 'rs', '₹', '$']
        invoice_matches = sum(1 for term in invoice_terms if term in text_lower)
        score += invoice_matches * 0.1
        
        return score
    
    def extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced line item extraction with medical focus"""
        if not text:
            return []
            
        lines = text.split('\n')
        line_items = []
        
        # Enhanced patterns for medical invoices
        patterns = [
            # Pattern: Medical Name Quantity Rate Amount
            r'([A-Za-z][A-Za-z\s]*(?:\d+[mg]|Tab|Cap|Syr|Inj|Tablet|Capsule|Syrup|Injection)[A-Za-z\s]*)\s+(\d+)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
            # Pattern: Name x Quantity @ Rate = Amount
            r'([A-Za-z][A-Za-z\s]*(?:\d+[mg]|Tab|Cap|Syr|Inj)[A-Za-z\s]*)\s+x\s*(\d+)\s*@\s*([\d,]+\.?\d*)\s*=\s*([\d,]+\.?\d*)',
            # Pattern: Name Rate Quantity Amount
            r'([A-Za-z][A-Za-z\s]*(?:\d+[mg]|Tab|Cap|Syr|Inj)[A-Za-z\s]*)\s+([\d,]+\.?\d*)\s+(\d+)\s+([\d,]+\.?\d*)',
            # Pattern: Simple medical item with amount
            r'([A-Za-z][A-Za-z\s]*(?:\d+[mg]|Tab|Cap|Syr|Inj)[A-Za-z\s]*)\s+([\d,]+\.\d{2})'
        ]
        
        for line in lines:
            line = line.strip()
            if not self._is_potential_line_item(line):
                continue
                
            item = self._extract_with_enhanced_patterns(line, patterns)
            if item and self._is_valid_medical_item(item):
                line_items.append(item)
        
        return line_items
    
    def _is_potential_line_item(self, line: str) -> bool:
        """Improved line item detection"""
        if not line or len(line) < 5:
            return False
            
        line_lower = line.lower()
        
        # Exclude headers and footers
        exclusion_terms = [
            'total', 'subtotal', 'tax', 'gst', 'vat', 'discount',
            'invoice', 'bill', 'receipt', 'date', 'time', 
            'patient', 'doctor', 'hospital', 'clinic',
            'phone', 'address', 'thank you', 'signature'
        ]
        
        if any(term in line_lower for term in exclusion_terms):
            return False
        
        # Must contain numbers (prices/quantities)
        if not re.search(r'\d+\.?\d*', line):
            return False
            
        return True
    
    def _extract_with_enhanced_patterns(self, line: str, patterns: List[str]) -> Optional[Dict[str, Any]]:
        """Enhanced pattern matching for medical items"""
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                if len(groups) == 2:
                    # Name + Amount
                    name, amount = groups
                    return {
                        "item_name": self._clean_medical_name(name),
                        "item_quantity": 1.0,
                        "item_rate": self._parse_number(amount),
                        "item_amount": self._parse_number(amount)
                    }
                elif len(groups) == 4:
                    name, val1, val2, amount = groups
                    
                    # Smart detection of quantity vs rate
                    if self._is_likely_quantity(val1) and self._is_likely_rate(val2):
                        quantity, rate = val1, val2
                    elif self._is_likely_rate(val1) and self._is_likely_quantity(val2):
                        rate, quantity = val1, val2
                    else:
                        # Default assumption
                        quantity, rate = val1, val2
                    
                    return {
                        "item_name": self._clean_medical_name(name),
                        "item_quantity": self._parse_number(quantity),
                        "item_rate": self._parse_number(rate),
                        "item_amount": self._parse_number(amount)
                    }
        
        return None
    
    def _is_likely_quantity(self, value: str) -> bool:
        """Check if value is likely a quantity"""
        try:
            num = float(value)
            return num == int(num) and 1 <= num <= 100
        except:
            return False
    
    def _is_likely_rate(self, value: str) -> bool:
        """Check if value is likely a rate/price"""
        try:
            num = float(value)
            return 0.5 <= num <= 5000
        except:
            return False
    
    def _clean_medical_name(self, name: str) -> str:
        """Enhanced medical name cleaning"""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Medical term standardization
        replacements = {
            'tab': 'Tab', 'capsule': 'Cap', 'syrup': 'Syr', 'injection': 'Inj',
            'tablet': 'Tab', 'cap': 'Cap', 'syr': 'Syr', 'inj': 'Inj'
        }
        
        words = name.split()
        cleaned_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in replacements:
                cleaned_words.append(replacements[word_lower])
            elif word_lower in ['mg', 'ml', 'gm', 'kg']:
                cleaned_words.append(word.upper())
            else:
                cleaned_words.append(word.capitalize())
        
        return ' '.join(cleaned_words)
    
    def _is_valid_medical_item(self, item: Dict[str, Any]) -> bool:
        """Enhanced validation for medical items"""
        name = item.get("item_name", "")
        quantity = item.get("item_quantity", 0)
        rate = item.get("item_rate", 0)
        amount = item.get("item_amount", 0)
        
        # Name validation
        if not name or len(name) < 3:
            return False
        
        # Check for medical terms in name
        name_lower = name.lower()
        has_medical_term = any(term in name_lower for term in self.medical_terms)
        if not has_medical_term:
            return False
        
        # Value validation
        if quantity <= 0 or rate <= 0 or amount <= 0:
            return False
        
        # Reasonable range checks
        if quantity > 100 or rate > 10000 or amount > 50000:
            return False
        
        # Amount validation with tolerance
        calculated = round(rate * quantity, 2)
        tolerance = max(1.0, amount * 0.1)  # 10% tolerance
        
        return abs(calculated - amount) <= tolerance
    
    def _parse_number(self, num_str: str) -> float:
        """Robust number parsing"""
        try:
            # Remove commas and non-numeric except decimal
            cleaned = re.sub(r'[^\d.]', '', num_str)
            if not cleaned:
                return 0.0
                
            # Handle multiple decimals
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = parts[0] + '.' + ''.join(parts[1:])
                
            return round(float(cleaned), 2)
        except (ValueError, TypeError):
            return 0.0
    
    def analyze_document(self, document_url: str) -> Dict[str, Any]:
        """Main analysis with confidence scoring"""
        try:
            # Download document
            response = requests.get(document_url, timeout=30)
            response.raise_for_status()
            document_content = response.content
            
            # Extract text
            text = self.extract_text_from_content(document_content)
            
            if not text:
                return self._get_fallback_data()
            
            # Extract line items
            line_items = self.extract_line_items(text)
            
            if not line_items:
                return self._get_fallback_data()
            
            # Calculate confidence
            confidence = self._calculate_confidence(text, line_items)
            total_amount = sum(item["item_amount"] for item in line_items)
            
            self.logger.info(f"Extraction successful: {len(line_items)} items, confidence: {confidence:.2f}")
            
            return {
                "line_items": line_items,
                "totals": {"Total": total_amount},
                "confidence": confidence,
                "raw_text": text[:300]
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return self._get_fallback_data()
    
    def _calculate_confidence(self, text: str, line_items: List[Dict]) -> float:
        """Calculate confidence based on multiple factors"""
        confidence = 0.5
        
        # Text quality
        text_score = self._score_text_quality(text)
        confidence += min(0.3, text_score * 0.1)
        
        # Number of valid items
        if line_items:
            confidence += min(0.2, len(line_items) * 0.05)
        
        # Item validation rate
        valid_items = sum(1 for item in line_items if self._is_valid_medical_item(item))
        if line_items:
            validation_rate = valid_items / len(line_items)
            confidence += validation_rate * 0.2
        
        return min(0.95, confidence)
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Enhanced fallback with medical focus"""
        try:
            from .mock_extractor import MockExtractor
            return MockExtractor().analyze_document(b"")
        except Exception as e:
            self.logger.error(f"Fallback failed: {e}")
            return {
                "line_items": [],
                "totals": {"Total": 0.0},
                "confidence": 0.0,
                "raw_text": ""
            }


def create_tesseract_extractor() -> TesseractExtractor:
    return TesseractExtractor()
