"""FastAPI Application Main Module for Recommendation Backend."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from ml.api import config as api_config
from ml.api.exceptions import register_exception_handlers
from ml.api.middleware import RequestLoggingMiddleware
from ml.api.routers import health, recommendations, movies, users, ratings
from ml.src.content_based import ContentBasedRecommender
from ml.src.collaborative_filtering import ItemBasedCollaborativeRecommender
from ml.src.hybrid_recommender import HybridRecommender

from ml.api.database.base import Base
from ml.api.database.session import engine

# Configure production-level logging format
log_level = getattr(logging, api_config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for validating config, DB tables, loading datasets and fitting models at application startup."""
    logger.info("Validating backend configuration...")
    try:
        api_config.validate_config()
    except Exception as e:
        logger.error("Configuration validation error: %s", str(e))
        if api_config.ENV == "production":
            raise e

    logger.info("Initializing database tables...")
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logger.warning("Could not auto-create database tables on startup: %s", str(e))

    logger.info("Initializing FastAPI recommendation models during lifespan startup...")
    try:
        movies_path = api_config.MOVIES_CSV_PATH
        ratings_path = api_config.RATINGS_CSV_PATH

        if not movies_path.exists() or not ratings_path.exists():
            logger.error("Processed CSV datasets missing at %s.", api_config.PROCESSED_DATA_DIR)
            app.state.content_recommender = None
            app.state.collaborative_recommender = None
            app.state.hybrid_recommender = None
            app.state.movies_df = None
            yield
            return

        logger.info("Loading processed datasets from %s...", api_config.PROCESSED_DATA_DIR)
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        app.state.movies_df = movies_df

        logger.info("Fitting Phase 3 Content-Based Recommender...")
        content_rec = ContentBasedRecommender()
        content_rec.fit(movies_df)
        app.state.content_recommender = content_rec

        logger.info("Fitting Phase 4 Item-Based Collaborative Filtering Recommender...")
        collab_rec = ItemBasedCollaborativeRecommender()
        collab_rec.fit(ratings_df, movies_df=movies_df)
        app.state.collaborative_recommender = collab_rec

        logger.info("Initializing Phase 5 Hybrid Recommender (50/50 default weights)...")
        hybrid_rec = HybridRecommender(
            content_recommender=content_rec,
            collaborative_recommender=collab_rec,
            content_weight=0.5,
            collaborative_weight=0.5
        )
        app.state.hybrid_recommender = hybrid_rec

        logger.info("All recommendation models fitted and ready successfully!")

    except Exception as e:
        logger.error("Failed to initialize recommendation models: %s", str(e), exc_info=True)
        app.state.content_recommender = None
        app.state.collaborative_recommender = None
        app.state.hybrid_recommender = None
        app.state.movies_df = None

    yield

    logger.info("Shutting down FastAPI application lifespan...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title=api_config.API_TITLE,
        description=api_config.API_DESCRIPTION,
        version=api_config.API_VERSION,
        openapi_url=f"{api_config.API_V1_STR}/openapi.json",
        docs_url="/docs" if api_config.DEBUG or api_config.ENV != "production" else None,
        redoc_url="/redoc" if api_config.DEBUG or api_config.ENV != "production" else None,
        lifespan=lifespan
    )

    # Register non-sensitive request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register centralized exception handlers
    register_exception_handlers(app)

    # Include API Routers under /api/v1
    app.include_router(health.router, prefix=api_config.API_V1_STR)
    app.include_router(recommendations.router, prefix=api_config.API_V1_STR)
    app.include_router(movies.router, prefix=api_config.API_V1_STR)
    app.include_router(users.router, prefix=api_config.API_V1_STR)
    app.include_router(ratings.router, prefix=api_config.API_V1_STR)

    return app


app = create_app()
