from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import logging
from src.extraction.pipeline import BillExtractionPipeline

logger = logging.getLogger(__name__)

# Initialize pipeline with mock mode for testing
# Set to False when you have real API keys
pipeline = BillExtractionPipeline(use_mock=True)

app = FastAPI(
    title="Medical Bill Extraction API",
    description="API for extracting line items from medical bills and invoices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BillRequest(BaseModel):
    document: HttpUrl

class BillResponse(BaseModel):
    is_success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "Medical Bill Extraction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "bill-extraction-api"}

@app.post("/extract-bill-data", response_model=BillResponse)
async def extract_bill_data(request: BillRequest):
    """
    Extract bill data from document URL
    
    - **document**: Publicly accessible URL of the bill document (image/PDF)
    """
    try:
        logger.info(f"Processing bill extraction request for: {request.document}")
        
        result = pipeline.process_document(str(request.document))
        
        if not result["is_success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        logger.info(f"Successfully processed request. Items found: {result['data']['total_item_count']}")
        return BillResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in API: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing document"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)