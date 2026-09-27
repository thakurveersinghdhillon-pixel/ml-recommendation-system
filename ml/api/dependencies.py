"""FastAPI Dependency Injection providers."""

from fastapi import Request, Depends
from ml.api.services.recommendation_service import RecommendationService


def get_recommendation_service(request: Request) -> RecommendationService:
    """Dependency injector providing initialized RecommendationService instance from app state."""
    content_rec = getattr(request.app.state, "content_recommender", None)
    collab_rec = getattr(request.app.state, "collaborative_recommender", None)
    hybrid_rec = getattr(request.app.state, "hybrid_recommender", None)
    movies_df = getattr(request.app.state, "movies_df", None)

    return RecommendationService(
        content_recommender=content_rec,
        collaborative_recommender=collab_rec,
        hybrid_recommender=hybrid_rec,
        movies_df=movies_df
    )
