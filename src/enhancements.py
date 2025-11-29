def enhanced_error_response(error_type, details=""):
    """Provide helpful error responses with guidance"""
    
    error_templates = {
        "invalid_input": {
            "status": "error",
            "error_code": "VALIDATION_001",
            "message": "Input validation failed",
            "details": details,
            "suggestion": "Please check the bill data structure and ensure required fields are present",
            "docs_url": "https://github.com/your-repo/docs/errors/VALIDATION_001"
        },
        "processing_error": {
            "status": "error", 
            "error_code": "PROCESS_002",
            "message": "Bill processing failed",
            "details": details,
            "suggestion": "Try simplifying the bill structure or check data format. Ensure amounts are properly formatted.",
            "docs_url": "https://github.com/your-repo/docs/errors/PROCESS_002"
        },
        "extraction_error": {
            "status": "error",
            "error_code": "EXTRACT_003", 
            "message": "Data extraction failed",
            "details": details,
            "suggestion": "Verify the bill contains clear line items with names and amounts",
            "docs_url": "https://github.com/your-repo/docs/errors/EXTRACT_003"
        }
    }
    
    return error_templates.get(error_type, {
        "status": "error",
        "error_code": "UNKNOWN_000",
        "message": "An unexpected error occurred",
        "details": details,
        "suggestion": "Please try again or check the documentation",
        "docs_url": "https://github.com/your-repo/docs"
    })
