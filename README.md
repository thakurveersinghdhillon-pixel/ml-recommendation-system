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
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

#### 2. Cosine Similarity
$$\text{Similarity}(\vec{A}, \vec{B}) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

---

## Phase 4 — Collaborative Filtering

Phase 4 implements an **Item-Based Collaborative Filtering Engine** (`ItemBasedCollaborativeRecommender`) using user-movie rating interaction history.

### What is Collaborative Filtering?
Collaborative filtering generates recommendations by analyzing collective user interaction behavior rather than item metadata. Item-Based Collaborative Filtering represents movies as rating vectors across users and identifies similar movies based on shared user rating patterns.

### Core Architectural Distinctions: Phase 3 vs. Phase 4

```text
Phase 3 (Content-Based):
Movie Metadata (title, genres) → TF-IDF Vectorization → Content Cosine Similarity → Content Recommendations

Phase 4 (Collaborative Filtering):
User Ratings (userId, movieId, rating) → Sparse User-Item Matrix → Rating Cosine Similarity → Normalized User Scoring → Personalized Recommendations
```

> **Metadata Independence**: Item-item collaborative similarity is calculated **strictly using user rating vectors** ($R_{u, i}$). Movie titles and genres are never used to compute collaborative similarity.

### Mathematical Methodology

#### 1. Sparse User-Item Matrix
The user interaction history is stored in a Scipy CSR sparse matrix $R \in \mathbb{R}^{U \times M}$ ($610 \text{ users} \times 9,742 \text{ movies}$). Unobserved interactions remain sparse zeros in memory. Duplicate `(userId, movieId)` pairs are aggregated deterministically using mean rating.

#### 2. Item-Item Cosine Similarity
Pairwise similarity between movie rating column vectors $\vec{v}_i$ and $\vec{v}_j$:
$$S_{i, j} = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\|_2 \|\vec{v}_j\|_2}$$
The full item-item similarity matrix $S \in \mathbb{R}^{9,742 \times 9,742}$ stored in `float32` requires $\approx \mathbf{380 \text{ MB}}$ of memory, which is lightweight and fast for the MovieLens-small dataset.

#### 3. Normalized Preference Scoring Formula
To generate recommendations for user $u$, the engine calculates normalized preference scores for all unrated candidate movies $c$:
$$\text{Score}(u, c) = \frac{\sum_{i \in \text{Rated}(u)} [S_{i, c} \times (r_{u, i} - \bar{r}_u)]}{\sum_{i \in \text{Rated}(u)} |S_{i, c}| + \epsilon}$$
* **Mean-Centering ($r_{u, i} - \bar{r}_u$)**: Ratings above user average $\bar{r}_u$ act as **positive preference signals**, while ratings below average act as **negative signals**.
* **Normalization ($\sum |S_{i, c}| + \epsilon$)**: Prevents bias toward candidates that have high similarity to many rated items regardless of user sentiment. $\epsilon = 10^{-9}$ prevents division by zero.

#### 4. Mandatory Already-Rated Filtering
When calling `recommend_for_user(user_id)`, all movies already rated by user $u$ in the training set are strictly filtered out from the final recommendations.

#### 5. Deterministic Candidate Ranking
- **Personalized User Recommendations**: Primary sort by `score` **DESCENDING**, secondary sort by `movieId` **ASCENDING**.
- **Similar Movie Recommendations**: Primary sort by `similarity_score` **DESCENDING**, secondary sort by `movieId` **ASCENDING**.

#### 6. Data Leakage Protection
The model relies **exclusively** on the DataFrame supplied to `fit(ratings_df, movies_df=None)` without internally reloading full datasets, enabling clean train/test evaluation in Phase 6.

### Cold-Start Limitations
* **New User Cold-Start**: A user with no historical ratings cannot receive personalized collaborative recommendations (`ValueError` is raised).
* **New Movie Cold-Start**: A movie with no user ratings has no collaborative vector and cannot be compared (`ValueError` is raised).
* *Note*: Cold-start limitations will be resolved in Phase 5 by combining content-based and collaborative signals in a Hybrid Recommendation System.

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
│   │   ├── config.py                   # Pathing & configuration management
│   │   ├── data_loader.py              # MovieLens dataset loading functions
│   │   ├── data_validator.py           # Dataset schema & integrity validation
│   │   ├── preprocessing.py            # Movie, rating, and genre preprocessing
│   │   ├── eda.py                      # Statistical aggregations & plot generation
│   │   ├── content_based.py            # Phase 3 Content-Based Recommender (TF-IDF & Cosine Similarity)
│   │   └── collaborative_filtering.py  # Phase 4 Item-Based Collaborative Filtering Recommender
│   ├── scripts/
│   │   ├── load_dataset.py             # Phase 1 loader & validator script
│   │   ├── run_eda.py                  # Phase 2 preprocessing & EDA pipeline script
│   │   ├── test_content_recommender.py # Phase 3 Content Recommender manual test script
│   │   └── test_collaborative_recommender.py # Phase 4 Collaborative Recommender manual test script
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py            # Unit tests for data loading (Phase 1)
│   │   ├── test_data_validator.py         # Unit tests for data validation (Phase 1)
│   │   ├── test_preprocessing.py          # Unit tests for preprocessing & EDA (Phase 2)
│   │   ├── test_content_based.py          # Unit tests for content recommender (Phase 3)
│   │   └── test_collaborative_filtering.py # Unit tests for collaborative recommender (Phase 4)
│   └── requirements.txt                # Pinned Python ML dependencies
├── docs/
├── .env.example                        # Example environment variables
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation
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

### 3. Run Collaborative Filtering Recommender Verification Script

```bash
python3 ml/scripts/test_collaborative_recommender.py
```

Loads processed ratings and movies, fits collaborative recommender, outputs top 10 personalized recommendations for User 1, displays item-item rating similarities, demonstrates collaborative explanations, and verifies error handling.

### 4. Run Full Automated Test Suite

```bash
python3 -m pytest ml/tests/
```

Executes all 69 unit tests across Phase 1, Phase 2, Phase 3, and Phase 4.
