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
* **Automated Test Suite**: Added 9 new unit tests verifying preprocessing, statistics calculation, matrix construction, zero-rating preservation, and sparsity math.

---

## Phase 3 — Content-Based Recommendation System

Phase 3 implements a production-quality, modular, testable, and explainable **Content-Based Movie Recommendation Engine** (`ContentBasedRecommender`).

### What is Content-Based Recommendation?
Content-based recommendation suggests items to users by evaluating similarity between item metadata features rather than relying on user rating behavior. In this system, recommendations are computed by comparing textual metadata representation of movies.

### Features Used
* **Movie Title**: Cleaned text tokens from movie titles.
* **Movie Genres**: Genre descriptors (`Adventure`, `Animation`, `Children`, `Comedy`, `Fantasy`, etc.).

### Mathematical Methodology

#### 1. TF-IDF (Term Frequency – Inverse Document Frequency)
TF-IDF converts unstructured movie metadata text into high-dimensional numerical feature vectors:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
* **Term Frequency ($\text{TF}$)**: Measures how frequently token $t$ appears in movie document $d$.
* **Inverse Document Frequency ($\text{IDF}$)**: Down-weights common words across all movies $D$ and emphasizes distinctive genre/title descriptors.

#### 2. Cosine Similarity
To evaluate similarity between two movie feature vectors $\vec{A}$ and $\vec{B}$, we calculate the cosine of the angle between them:
$$\text{Similarity}(\vec{A}, \vec{B}) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
Cosine similarity ranges from `0.0` (completely orthogonal/dissimilar) to `1.0` (identical content direction).

### Recommendation Pipeline
```text
Movie Catalog Metadata (title, genres)
          ↓
Content Text Representation ("toy story adventure animation children comedy fantasy")
          ↓
TF-IDF Vectorization (scikit-learn TfidfVectorizer)
          ↓
Pairwise Cosine Similarity Matrix (9,742 x 9,742)
          ↓
Title Normalization & Lookup
          ↓
Exclude Query Source Movie & Rank Candidates (Similarity DESC, movieId ASC)
          ↓
Top-K Recommendation Items & Metadata-Grounded Explanations
```

### Key Accomplishments in Phase 3
* **Modular Recommender Module**: Implemented `ContentBasedRecommender` in `ml/src/content_based.py` with `fit()`, `get_movie_index()`, `get_similarity()`, `recommend()`, and `explain_recommendation()`.
* **Deterministic Title Lookup & Duplicates**: Normalizes case and whitespace; deterministically resolves duplicate titles (first occurrence in catalog wins).
* **Metadata-Grounded Explanations**: Generates factual, deterministic explanations reporting shared genres and actual cosine similarity scores (no LLM dependence).
* **Robust Edge Case Handling**: Explicit validation for unknown movies, empty titles, invalid $N \le 0$, missing/empty genres, and $N > \text{catalog size}$.
* **Comprehensive Test Suite**: Added 23 unit tests in `ml/tests/test_content_based.py` (total 45 tests passing).

### Advantages & Limitations of Content-Based Filtering

#### Advantages:
1. **Solves Item Cold-Start**: Can immediately recommend new or unrated movies as long as metadata (genres, title) is available.
2. **High Interpretability**: Explanations can explicitly point to shared attributes (e.g. shared genres `"Adventure"`, `"Animation"`).
3. **No User Inter-dependency**: Recommendations do not depend on other users' interaction data.

#### Limitations:
1. **Over-Specialization**: Tends to recommend items very similar to the query movie (lack of serendipity).
2. **Metadata Dependency**: Quality of recommendations is bound by the richness and accuracy of metadata.
3. **Ignores Collaborative Signals**: Does not capture user preference patterns or rating sentiment.

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
│   │   ├── eda.py              # Statistical aggregations & plot generation
│   │   └── content_based.py    # Content-Based Recommender (TF-IDF & Cosine Similarity)
│   ├── scripts/
│   │   ├── load_dataset.py     # Phase 1 loader & validator script
│   │   ├── run_eda.py          # Phase 2 preprocessing & EDA pipeline script
│   │   └── test_content_recommender.py # Phase 3 Content Recommender manual test script
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py    # Unit tests for data loading (Phase 1)
│   │   ├── test_data_validator.py # Unit tests for data validation (Phase 1)
│   │   ├── test_preprocessing.py  # Unit tests for preprocessing & EDA (Phase 2)
│   │   └── test_content_based.py  # Unit tests for content recommender (Phase 3)
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

### 3. Run Content-Based Recommender Verification Script

```bash
python3 ml/scripts/test_content_recommender.py
```

Loads processed movies, fits TF-IDF recommender, outputs top 10 recommendations for `"Toy Story (1995)"`, and demonstrates explanations and error handling.

### 4. Run Full Automated Test Suite

```bash
python3 -m pytest ml/tests/
```

Executes all 45 unit tests across Phase 1, Phase 2, and Phase 3.
