"""Movie domain Pydantic schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class MovieResponse(BaseModel):
    """Movie metadata item schema."""
    movie_id: int = Field(..., description="Unique MovieLens Movie ID.", examples=[1])
    title: str = Field(..., description="Movie title string.", examples=["Toy Story (1995)"])
    genres: List[str] = Field(..., description="List of genre labels.", examples=[["Adventure", "Animation", "Children", "Comedy", "Fantasy"]])
    year: Optional[int] = Field(None, description="Movie release year if available.", examples=[1995])


class MovieSearchResponse(BaseModel):
    """Movie search response schema."""
    query: str = Field(..., description="Search query string.", examples=["toy"])
    total_matches: int = Field(..., description="Total matching movies found.", examples=[3])
    limit: int = Field(..., description="Maximum results limit returned.", examples=[10])
    movies: List[MovieResponse] = Field(..., description="Matching movie metadata items.")
