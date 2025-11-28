import requests
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging
from typing import Optional, Tuple
import tempfile
import os

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, max_file_size_mb: int = 10, timeout: int = 30):
        self.session = requests.Session()
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.timeout = timeout
        self.session.headers.update({
            'User-Agent': 'BillExtractionBot/1.0'
        })
    
    def download_document(self, document_url: str) -> Optional[bytes]:
        """Download document from URL with error handling"""
        try:
            # Validate URL
            if not document_url.startswith(('http://', 'https://')):
                raise ValueError("Invalid URL protocol")
            
            # Head request to check file size
            head_response = self.session.head(document_url, timeout=10, allow_redirects=True)
            head_response.raise_for_status()
            
            content_length = head_response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size:
                raise ValueError(f"File too large: {content_length} bytes")
            
            # Download document
            response = self.session.get(document_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image', 'pdf', 'octet-stream']):
                logger.warning(f"Unexpected content type: {content_type}")
            
            # Read content
            content = response.content
            
            if len(content) > self.max_file_size:
                raise ValueError(f"Downloaded file exceeds size limit: {len(content)} bytes")
            
            logger.info(f"Successfully downloaded document: {len(content)} bytes")
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading document: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading document: {e}")
            return None
    
    def preprocess_image(self, image_content: bytes) -> Tuple[bytes, str]:
        """Enhance image quality for better OCR and return processed image"""
        try:
            # Detect image format
            image = Image.open(io.BytesIO(image_content))
            original_format = image.format
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to OpenCV for processing
            cv_image = np.array(image)
            
            # Apply preprocessing pipeline
            enhanced = self._enhance_image(cv_image)
            
            # Convert back to bytes
            enhanced_pil = Image.fromarray(enhanced)
            output_buffer = io.BytesIO()
            enhanced_pil.save(output_buffer, format=original_format or 'JPEG', quality=95)
            
            return output_buffer.getvalue(), original_format or 'JPEG'
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original if processing fails
            return image_content, 'JPEG'
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement"""
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Step 1: Noise removal
        denoised = cv2.medianBlur(gray, 3)
        
        # Step 2: Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Step 3: Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        
        return sharpened
    
    def validate_document(self, content: bytes) -> bool:
        """Validate document before processing"""
        try:
            # Check file size
            if len(content) > self.max_file_size:
                return False
            
            # Try to open as image
            image = Image.open(io.BytesIO(content))
            image.verify()  # Verify it's a valid image
            
            # Check dimensions
            width, height = image.size
            if width < 100 or height < 100:
                logger.warning("Image dimensions too small")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Document validation failed: {e}")
            return False