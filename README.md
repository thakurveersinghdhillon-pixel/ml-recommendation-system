# Hybrid ML Recommendation System

A production-style **Hybrid ML Recommendation System** built with Python, Pandas, NumPy, Scikit-learn, FastAPI, PostgreSQL, Next.js, TypeScript, and Docker.

---

## Phase 1 — Dataset & ML Environment Setup

Phase 1 establishes a reproducible Python ML foundation and data loading/validation pipeline for the MovieLens dataset (`ml-latest-small`).

### Key Accomplishments in Phase 1
* **Modular Project Architecture**: Separated configuration, loading logic, validation logic, scripts, and tests.
* **Idempotent Data Pipeline**: Automatic download and extraction of the MovieLens dataset into `ml/data/raw/` without unnecessary re-downloading.
* **Schema & Integrity Validation**: Checks for required columns, null keys, rating score ranges, duplicates, and referential integrity between ratings and movie catalog.
* **CLI Dataset Statistics**: `load_dataset.py` script computes and reports dataset statistics and data quality metrics.
* **Automated Unit Testing**: Comprehensive test suite using `pytest` with sample data fixtures.

---

## Phase 2 — Data Preprocessing and EDA

Phase 2 focuses on data cleaning, transformation, statistical summary computation, interaction matrix construction, sparsity calculation, and visualization generation to prepare the dataset for recommendation algorithms.

### Key Accomplishments in Phase 2
* **Movie Preprocessing**: Extracted 4-digit release year from title strings into a dedicated `year` column while preserving original titles and unrated catalog movies.
* **Genre Parsing**: Transformed pipe-separated genre strings into clean list structures, explicitly handling `(no genres listed)` without errors.
* **Rating Datetime Conversion**: Converted Unix timestamps to Pandas `datetime` objects for time-series activity analysis.
* **Preservation of Zero-Rating Movies**: Used a `LEFT JOIN` from movie catalog to rating aggregations to retain unrated movies with `rating_count = 0` and `average_rating = NaN` (avoiding fake 0.0 rating assignments).
* **Exploded Genre Analytics**: Computed per-genre movie counts, total rating counts, and average ratings without double-counting unique movies.
* **User-Item Matrix & Sparsity**: Constructed a pivoted user-item matrix (`610` users x `9,742` movies) preserving `NaN` for unobserved interactions. Computed exact matrix sparsity (**98.30%**).
* **EDA Visualizations**: Generated 8 publication-quality plots in `ml/data/processed/eda/`.
* **Automated Test Suite**: Added 9 new unit tests (22 total tests passing) verifying preprocessing, statistics calculation, matrix construction, zero-rating preservation, and sparsity math.

---

## Dataset Findings & Summary Statistics

| Metric | Calculated Value |
| :--- | :--- |
| **Total Users** | `610` |
| **Catalog Movies** | `9,742` |
| **Observed Ratings** | `100,836` |
| **Zero-Rating Movies** | `18` |
| **Mean Rating** | `3.50` |
| **Median Rating** | `3.50` |
| **Rating Std Deviation** | `1.04` |
| **Average Ratings per User** | `165.30` |
| **Average Ratings per Movie** | `10.35` |
| **Unique Genres** | `20` |
| **Possible Interactions** | `5,942,620` |
| **Matrix Sparsity** | **`98.30%`** |
| **Matrix Density** | **`1.70%`** |

### Why Sparsity Matters for Collaborative Filtering
A matrix sparsity of **98.30%** means that **98.3% of all possible user-movie interactions are unobserved (missing)**. Only 1.70% of potential ratings exist in the dataset. This high degree of sparsity is a fundamental characteristic of recommendation systems. It highlights why naive similarity calculations (e.g. raw dot products) fail on unobserved entries, and motivates model-based collaborative filtering (such as Matrix Factorization / SVD) and hybrid content-based filtering to address the "cold-start" problem.

---

## Directory Structure

```text
ML project/
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   │   └── ml-latest-small/
│   │   │       ├── movies.csv
│   │   │       └── ratings.csv
│   │   └── processed/
│   │       ├── movies_processed.csv
│   │       ├── ratings_processed.csv
│   │       ├── user_statistics.csv
│   │       ├── movie_statistics.csv
│   │       ├── genre_statistics.csv
│   │       └── eda/
│   │           ├── rating_distribution.png
│   │           ├── ratings_per_user_distribution.png
│   │           ├── ratings_per_movie_distribution.png
│   │           ├── top_popular_movies.png
│   │           ├── genre_distribution.png
│   │           ├── rating_vs_rating_count_scatter.png
│   │           ├── rating_activity_over_time.png
│   │           └── user_item_matrix_sparsity.png
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py           # Pathing & configuration management
│   │   ├── data_loader.py      # MovieLens dataset loading functions
│   │   ├── data_validator.py   # Dataset schema & integrity validation
│   │   ├── preprocessing.py    # Movie, rating, and genre preprocessing
│   │   └── eda.py              # Statistical aggregations & plot generation
│   ├── scripts/
│   │   ├── load_dataset.py     # Phase 1 loader & validator script
│   │   └── run_eda.py          # Phase 2 preprocessing & EDA pipeline script
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py    # Unit tests for data loading
│   │   ├── test_data_validator.py # Unit tests for data validation
│   │   └── test_preprocessing.py  # Unit tests for preprocessing & EDA
│   └── requirements.txt        # Pinned Python ML dependencies
├── docs/
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Quickstart Instructions

### 1. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r ml/requirements.txt
```

### 3. Run Data Preprocessing & EDA Pipeline

```bash
python3 ml/scripts/run_eda.py
```

Outputs processed CSVs into `ml/data/processed/` and 8 plots into `ml/data/processed/eda/`.

### 4. Run Test Suite

```bash
python3 -m pytest ml/tests/
```

Executes all 22 unit tests across data loading, schema validation, preprocessing, user/movie statistics, interaction matrix construction, and sparsity math.
