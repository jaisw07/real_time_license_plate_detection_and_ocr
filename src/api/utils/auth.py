from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "your_placeholder_secret_key_here")
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="token", auto_error=False)

async def get_api_key(
    header_key: str = Security(api_key_header),
    query_key: str = Security(api_key_query),
):
    if header_key == API_KEY or query_key == API_KEY:
        return True
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )
