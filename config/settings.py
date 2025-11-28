import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Azure Form Recognizer
    AZURE_FORM_RECOGNIZER_ENDPOINT = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    AZURE_FORM_RECOGNIZER_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
    
    # AWS Textract (Fallback)
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # LLM APIs
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def use_mock(self):
        """Use mock extractor if no Azure credentials"""
        return not (self.AZURE_FORM_RECOGNIZER_ENDPOINT and self.AZURE_FORM_RECOGNIZER_KEY)
    
    def validate(self):
        """Validate required settings"""
        if not self.AZURE_FORM_RECOGNIZER_ENDPOINT:
            raise ValueError("AZURE_FORM_RECOGNIZER_ENDPOINT is required")
        if not self.AZURE_FORM_RECOGNIZER_KEY:
            raise ValueError("AZURE_FORM_RECOGNIZER_KEY is required")

settings = Settings()
settings.validate()