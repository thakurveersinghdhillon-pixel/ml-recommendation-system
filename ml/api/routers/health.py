"""Health Check Router."""

from fastapi import APIRouter, Request, Response, status
from ml.api.schemas.common import HealthResponse, ModelHealthStatus, ErrorResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": HealthResponse, "description": "API service and models are healthy and ready."},
        503: {"model": ErrorResponse, "description": "Service unavailable or models not initialized."}
    }
)
async def health_check(request: Request, response: Response):
    """Verify that API is running and recommendation models are initialized and ready."""
    content_rec = getattr(request.app.state, "content_recommender", None)
    collab_rec = getattr(request.app.state, "collaborative_recommender", None)
    hybrid_rec = getattr(request.app.state, "hybrid_recommender", None)

    c_ready = content_rec is not None and getattr(content_rec, "is_fitted", False)
    cf_ready = collab_rec is not None and getattr(collab_rec, "is_fitted", False)
    h_ready = hybrid_rec is not None and c_ready and cf_ready

    c_status = "ready" if c_ready else "not_ready"
    cf_status = "ready" if cf_ready else "not_ready"
    h_status = "ready" if h_ready else "not_ready"

    all_ready = c_ready and cf_ready and h_ready

    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="unhealthy",
            service="movie-recommendation-api",
            models=ModelHealthStatus(
                content_based=c_status,
                collaborative=cf_status,
                hybrid=h_status
            )
        )

    return HealthResponse(
        status="healthy",
        service="movie-recommendation-api",
        models=ModelHealthStatus(
            content_based=c_status,
            collaborative=cf_status,
            hybrid=h_status
        )
    )
