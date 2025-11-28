import requests
import logging
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Optional, Tuple, Dict, Any

class DocumentProcessor:
    def __init__(self, max_file_size_mb: int = 10, timeout: int = 30):
        self.logger = logging.getLogger(__name__)
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.timeout = timeout
        
        # Create session with better headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MedicalBillExtractor/1.0'
        })
    
    def download_document(self, document_url: str) -> Optional[bytes]:
        """Download document from URL with enhanced error handling"""
        try:
            # Validate URL format
            if not document_url.startswith(('http://', 'https://')):
                self.logger.error("Invalid URL protocol")
                return None
            
            # Download with timeout and size limits
            response = self.session.get(
                document_url, 
                timeout=self.timeout, 
                stream=True,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Read content with size limit
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.max_file_size:
                    self.logger.error(f"File exceeds size limit: {len(content)} bytes")
                    return None
            
            # Basic content validation
            if len(content) < 100:  # Too small to be a valid image
                self.logger.warning("Downloaded content too small")
                return None
            
            # Check if it's a valid image
            if not self._is_valid_image(content):
                self.logger.warning("Downloaded content is not a valid image")
                return None
            
            self.logger.info(f"Successfully downloaded document: {len(content)} bytes")
            return content
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error downloading document: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading document: {e}")
            return None
    
    def _is_valid_image(self, content: bytes) -> bool:
        """Check if content is a valid image"""
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()  # Verify without loading entire image
            return True
        except Exception:
            return False
    
    def validate_document(self, document_content: bytes) -> bool:
        """Enhanced document validation with multiple checks"""
        if not document_content:
            return False
        
        # Size validation
        if len(document_content) < 100 or len(document_content) > self.max_file_size:
            self.logger.warning(f"Invalid document size: {len(document_content)} bytes")
            return False
        
        # Image format validation
        try:
            image = Image.open(io.BytesIO(document_content))
            image.verify()
            
            # Check dimensions
            width, height = image.size
            if width < 50 or height < 50:
                self.logger.warning(f"Image dimensions too small: {width}x{height}")
                return False
            if width > 10000 or height > 10000:
                self.logger.warning(f"Image dimensions too large: {width}x{height}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Document validation failed: {e}")
            return False
    
    def preprocess_image(self, document_content: bytes) -> Tuple[Any, Dict[str, Any]]:
        """Enhanced image preprocessing for better OCR"""
        try:
            # Open and verify image
            image = Image.open(io.BytesIO(document_content))
            original_format = image.format or 'JPEG'
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Store original info
            original_size = image.size
            original_mode = image.mode
            
            # Apply preprocessing pipeline
            processed_image = self._enhance_image_for_ocr(image)
            
            # Return both processed image and metadata
            metadata = {
                "format": original_format,
                "original_size": original_size,
                "original_mode": original_mode,
                "processed_size": processed_image.size,
                "processed": True,
                "enhancements_applied": True
            }
            
            return processed_image, metadata
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed, returning original: {e}")
            # Return original content with basic metadata
            return document_content, {
                "format": "unknown", 
                "processed": False,
                "error": str(e)
            }
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements specifically optimized for OCR"""
        try:
            # Step 1: Resize if too small (better for OCR)
            width, height = image.size
            if width < 800:
                # Calculate new dimensions maintaining aspect ratio
                scale_factor = 1600 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Step 2: Convert to grayscale for better OCR (optional but often improves accuracy)
            # image = image.convert('L').convert('RGB')  # Uncomment if grayscale improves your OCR
            
            # Step 3: Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.8)  # Moderate contrast boost
            
            # Step 4: Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.6)  # Moderate sharpness boost
            
            # Step 5: Reduce noise with median filter
            image = image.filter(ImageFilter.MedianFilter(3))
            
            # Step 6: Optional brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)  # Slight brightness boost
            
            self.logger.info("Image preprocessing completed successfully")
            return image
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image  # Return original if enhancement fails
    
    def get_document_info(self, document_content: bytes) -> Dict[str, Any]:
        """Get detailed information about the document"""
        try:
            image = Image.open(io.BytesIO(document_content))
            
            return {
                "size_bytes": len(document_content),
                "format": image.format,
                "dimensions": image.size,
                "mode": image.mode,
                "is_valid": True
            }
        except Exception as e:
            return {
                "size_bytes": len(document_content),
                "is_valid": False,
                "error": str(e)
            }
    
    def safe_preprocess(self, document_content: bytes) -> Tuple[Any, Dict[str, Any]]:
        """Safe preprocessing that never crashes"""
        try:
            return self.preprocess_image(document_content)
        except Exception as e:
            self.logger.error(f"Safe preprocessing failed: {e}")
            return document_content, {
                "format": "unknown",
                "processed": False,
                "error": "Preprocessing failed"
            }


# Factory function for easy initialization
def create_document_processor(max_file_size_mb: int = 10) -> DocumentProcessor:
    """Create and return a DocumentProcessor instance"""
    return DocumentProcessor(max_file_size_mb=max_file_size_mb)
