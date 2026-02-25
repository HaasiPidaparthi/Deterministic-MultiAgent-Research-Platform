from typing import List, Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None 

class FetchResult(BaseModel):
    url: str
    status_code: int
    title: Optional[str] = None
    publisher: Optional[str] = None
    text: str