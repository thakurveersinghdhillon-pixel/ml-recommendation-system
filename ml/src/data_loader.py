"""Data loader module for the MovieLens dataset."""

import io
import logging
from pathlib import Path
import zipfile
import pandas as pd
import requests

from ml.src import config

logger = logging.getLogger(__name__)


def find_dataset_file(file_name: str, data_dir: Path | str | None = None) -> Path:
    """Locate a dataset file within the raw data directory or any subfolder.

    Args:
        file_name: Target filename (e.g., 'movies.csv', 'ratings.csv').
        data_dir: Directory to search in. Defaults to config.RAW_DATA_DIR.

    Returns:
        Path to the found file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    target_dir = Path(data_dir) if data_dir is not None else config.RAW_DATA_DIR

    if not target_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {target_dir}")

    # Direct match in data_dir
    direct_path = target_dir / file_name
    if direct_path.is_file():
        return direct_path

    # Check immediate subdirectories (e.g., raw/ml-latest-small/movies.csv)
    matches = list(target_dir.rglob(file_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Required dataset file '{file_name}' not found in '{target_dir}'. "
        "Please download the dataset or set auto_download=True."
    )


def download_movielens_dataset(
    data_dir: Path | str | None = None,
    url: str | None = None
) -> Path:
    """Download and extract the MovieLens dataset if not already present (idempotent).

    Args:
        data_dir: Directory to extract files into. Defaults to config.RAW_DATA_DIR.
        url: URL of the MovieLens zip file. Defaults to config.MOVIELENS_URL.

    Returns:
        Path to the directory containing extracted files.
    """
    target_dir = Path(data_dir) if data_dir is not None else config.RAW_DATA_DIR
    download_url = url or config.MOVIELENS_URL

    target_dir.mkdir(parents=True, exist_ok=True)

    # Check idempotency: if both movies.csv and ratings.csv exist, skip downloading
    try:
        find_dataset_file("movies.csv", target_dir)
        find_dataset_file("ratings.csv", target_dir)
        logger.info("Dataset files already exist in %s. Skipping download.", target_dir)
        return target_dir
    except FileNotFoundError:
        logger.info("Dataset files missing. Downloading from %s...", download_url)

    response = requests.get(download_url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(target_dir)

    logger.info("Dataset successfully downloaded and extracted to %s.", target_dir)
    return target_dir


def load_movies(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load movies dataset into a Pandas DataFrame.

    Args:
        data_dir: Directory containing movies.csv. Defaults to config.RAW_DATA_DIR.

    Returns:
        Pandas DataFrame containing movie data.

    Raises:
        FileNotFoundError: If movies.csv is not found.
        ValueError: If required columns are missing.
    """
    file_path = find_dataset_file("movies.csv", data_dir)
    df = pd.read_csv(file_path)

    missing_cols = [col for col in config.REQUIRED_MOVIE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Movies dataset at {file_path} is missing required columns: {missing_cols}"
        )

    return df


def load_ratings(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load ratings dataset into a Pandas DataFrame.

    Args:
        data_dir: Directory containing ratings.csv. Defaults to config.RAW_DATA_DIR.

    Returns:
        Pandas DataFrame containing rating data.

    Raises:
        FileNotFoundError: If ratings.csv is not found.
        ValueError: If required columns are missing.
    """
    file_path = find_dataset_file("ratings.csv", data_dir)
    df = pd.read_csv(file_path)

    missing_cols = [col for col in config.REQUIRED_RATING_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Ratings dataset at {file_path} is missing required columns: {missing_cols}"
        )

    return df


def load_dataset(
    data_dir: Path | str | None = None,
    auto_download: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both movies and ratings datasets.

    Args:
        data_dir: Directory containing raw data files. Defaults to config.RAW_DATA_DIR.
        auto_download: If True, automatically download dataset if missing.

    Returns:
        Tuple of (movies_df, ratings_df).
    """
    target_dir = Path(data_dir) if data_dir is not None else config.RAW_DATA_DIR

    if auto_download:
        download_movielens_dataset(data_dir=target_dir)

    movies_df = load_movies(data_dir=target_dir)
    ratings_df = load_ratings(data_dir=target_dir)

    return movies_df, ratings_df
