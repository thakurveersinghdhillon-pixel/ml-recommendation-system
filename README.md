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
│   │   ├── collaborative_filtering.py  # Phase 4 Item-Based Collaborative Filtering Recommender
│   │   ├── hybrid_recommender.py       # Phase 5 Hybrid Recommendation Engine
│   │   ├── evaluation_utils.py         # Phase 6 Temporal split & ranking metrics
│   │   └── evaluation.py               # Phase 6 ModelEvaluator & visualization pipeline
│   ├── scripts/
│   │   ├── load_dataset.py             # Phase 1 loader & validator script
│   │   ├── run_eda.py                  # Phase 2 preprocessing & EDA pipeline script
│   │   ├── test_content_recommender.py # Phase 3 Content Recommender manual test script
│   │   ├── test_collaborative_recommender.py # Phase 4 Collaborative Recommender manual test script
│   │   ├── test_hybrid_recommender.py  # Phase 5 Hybrid Recommender manual test script
│   │   └── run_evaluation.py           # Phase 6 Model Evaluation pipeline script
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py            # Unit tests for data loading (Phase 1)
│   │   ├── test_data_validator.py         # Unit tests for data validation (Phase 1)
│   │   ├── test_preprocessing.py          # Unit tests for preprocessing & EDA (Phase 2)
│   │   ├── test_content_based.py          # Unit tests for content recommender (Phase 3)
│   │   ├── test_collaborative_filtering.py # Unit tests for collaborative recommender (Phase 4)
│   │   ├── test_hybrid_recommender.py     # Unit tests for hybrid recommender (Phase 5)
│   │   ├── test_evaluation.py             # Unit tests for evaluation framework (Phase 6)
│   │   ├── test_database.py               # Phase 8 ORM unit tests
│   │   ├── test_database_integration.py   # Phase 8 DB integration tests
│   │   ├── test_ratings.py                # Phase 8 Rating repository tests
│   │   ├── test_recommendation_history.py # Phase 8 Recommendation history tests
│   │   └── test_users.py                  # Phase 8 User CRUD tests
│   └── requirements.txt                # Pinned Python ML & Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── app/                        # Next.js App Router routes & pages
│   │   │   ├── layout.tsx              # Root layout & global providers
│   │   │   ├── page.tsx                # Home dashboard page
│   │   │   ├── globals.css             # Tailwind CSS global styles
│   │   │   ├── movies/
│   │   │   │   ├── page.tsx            # Catalog movie search page
│   │   │   │   └── [movieId]/page.tsx  # Movie details & similar movies
│   │   │   ├── recommendations/
│   │   │   │   └── page.tsx            # Interactive recommendation page (3 modes)
│   │   │   └── users/
│   │   │       ├── page.tsx            # User management page
│   │   │       └── [userId]/page.tsx  # User profile & rating history page
│   │   ├── components/                 # Reusable UI components
│   │   │   ├── layout/                 # Navbar & Footer components
│   │   │   ├── movies/                 # MovieCard component
│   │   │   ├── ratings/                # StarRating component
│   │   │   └── ui/                     # SkeletonCard & ConfirmationModal
│   │   ├── context/                    # UserContext & ToastContext providers
│   │   ├── lib/api/                    # Typed API client functions
│   │   ├── test/                       # Vitest & React Testing Library tests
│   │   └── types/                      # TypeScript interface schemas
│   ├── package.json                    # Frontend dependencies & scripts
│   ├── tsconfig.json                   # TypeScript configuration
│   ├── tailwind.config.ts              # Tailwind CSS configuration
│   ├── vitest.config.ts                # Vitest test framework configuration
│   └── .env.local                      # Local environment configuration
├── docs/
├── .env.example                        # Example environment variables
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation
```

---

## Phase 6 — Recommendation Model Evaluation Framework

Phase 6 implements a leakage-free, reproducible evaluation framework comparing Content-Based (Phase 3), Item-Based Collaborative Filtering (Phase 4), and Hybrid Recommendation (Phase 5) models on the MovieLens dataset.

### 1. Leakage-Free Temporal Split Methodology
- Ratings for each user are sorted chronologically (`timestamp ASC`, tie-breaker `movieId ASC`).
- Earlier interactions ($(1 - \text{test\_ratio})$) form `train_df`, later interactions ($\text{test\_ratio}=0.20$) form `test_df`.
- **Strict Data Isolation**: All models are fitted exclusively on `train_df`. `test_df` interactions are used solely for ground-truth relevance measurement.
- **Content Seed Strategy**: Content-based seed titles are selected from the user's most recent highly-rated training interaction ONLY (`timestamp DESC`, `rating DESC`, `movieId ASC`). Never from `test_df`.

### 2. Evaluated Ranking Metrics
- **Precision@K**: $\frac{|\text{Top-}K \cap \text{Relevant Test Set}|}{K}$
- **Recall@K**: $\frac{|\text{Top-}K \cap \text{Relevant Test Set}|}{|\text{Relevant Test Set}|}$ (Evaluated on users with $\ge 1$ relevant test item).
- **NDCG@K**: $\frac{\text{DCG}@K}{\text{IDCG}@K}$ where $\text{DCG}@K = \sum_{i=1}^K \frac{rel_i}{\log_2(i+1)}$.
- **Hit Rate@K**: $1.0$ if $|\text{Top-}K \cap \text{Relevant Test Set}| > 0$ else $0.0$.
- **Catalog Coverage**: Proportion of unique catalog movie IDs recommended across all evaluated users.
- **Relevance Threshold**: Ratings $\ge 4.0$ in test set.

### 3. Rating Prediction Metrics (RMSE / MAE) Non-Applicability
RMSE and MAE measure 1–5 star rating prediction error ($\hat{r}_{u,i}$). Because Phase 3, 4, and 5 compute relative similarity and ranking scores rather than absolute 1–5 star ratings, RMSE and MAE are mathematically non-applicable and excluded to maintain evaluation integrity.

---

## Phase 7 — FastAPI Backend Development

Phase 7 wraps the existing Phase 3 Content-Based, Phase 4 Collaborative Filtering, and Phase 5 Hybrid recommendation models into a production-oriented REST API built with FastAPI, Pydantic validation, lifespan model fitting, and OpenAPI/Swagger UI.

### 1. REST API Endpoints
- **`GET /api/v1/health`**: Service health check & model readiness report.
- **`POST /api/v1/recommendations/content`**: Content-based recommendations by seed `movie_id`.
- **`POST /api/v1/recommendations/collaborative`**: Collaborative recommendations by `user_id`.
- **`POST /api/v1/recommendations/hybrid`**: Hybrid recommendations by `user_id` and optional seed `movie_id`.
- **`GET /api/v1/movies/search?q={query}&limit={limit}`**: Partial title movie search.
- **`GET /api/v1/movies/{movie_id}`**: Movie details lookup.

### 2. Swagger UI & OpenAPI Docs
Interactive API documentation is served automatically at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **OpenAPI JSON**: `http://127.0.0.1:8000/api/v1/openapi.json`

---

## Quickstart Instructions

### 1. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r ml/requirements.txt
```

### 3. Start FastAPI Recommendation REST API Server (Phase 7)

```bash
uvicorn ml.api.main:app --reload --port 8000
```

Access Swagger UI interactive docs at `http://127.0.0.1:8000/docs`.

### 4. Run Offline Evaluation Pipeline (Phase 6)

```bash
python3 ml/scripts/run_evaluation.py
```

### 5. Run Full Automated Test Suite

```bash
python3 -m pytest ml/tests/
```

Executes all 148 unit and integration tests across Phase 1 through Phase 8.

---

## Phase 8 — PostgreSQL Database Integration

Phase 8 integrates PostgreSQL with SQLAlchemy 2.x ORM, Alembic migrations, and Pydantic v2 schemas to persist users, ratings, and recommendation request history while preserving all Phase 1–7 ML engines intact.

### 1. Database Schema & Models
* **`User` (`users`)**: Stores registered users (`id`, `username`, `email`, UTC timestamps).
* **`Rating` (`ratings`)**: Stores movie ratings (`id`, `user_id`, `movie_id`, `rating`, UTC timestamps). Enforces unique `(user_id, movie_id)` constraint and rating bounds ($0.5 \le rating \le 5.0$).
* **`RecommendationHistory` (`recommendation_history`)**: Persists successful recommendation request payloads and generated movie items (`id`, `user_id`, `model`, `seed_movie_id`, `top_k`, `content_weight`, `collaborative_weight`, `created_at`, `recommendations` JSON/JSONB).

### 2. REST API Endpoints (Phase 8 Additions)
* **`POST /api/v1/users`**: Register user.
* **`GET /api/v1/users/{user_id}`**: Fetch user details.
* **`PATCH /api/v1/users/{user_id}`**: Update user details.
* **`DELETE /api/v1/users/{user_id}`**: Delete user (cascades ratings/history).
* **`PUT /api/v1/ratings`**: Create or update user movie rating (atomic upsert).
* **`GET /api/v1/users/{user_id}/ratings`**: List paginated user ratings.
* **`GET /api/v1/users/{user_id}/ratings/{movie_id}`**: Fetch specific rating.
* **`DELETE /api/v1/users/{user_id}/ratings/{movie_id}`**: Delete rating.
* **`GET /api/v1/users/{user_id}/recommendation-history`**: Query logged recommendation requests.
* **`GET /api/v1/health`**: Extended to check DB connectivity (`SELECT 1`).

---

## Phase 9 — Frontend Development and API Integration

Phase 9 implements a full-featured, responsive, production-ready Next.js web interface integrated with the FastAPI backend and PostgreSQL database layer.

### 1. Core Pages & UI Components
* **Home Page (`/`)**: Highlighting system capabilities, fast search access, quick recommendation mode shortcuts, and backend status monitor.
* **Movie Discovery Catalog (`/movies`)**: Instant movie search with debounced inputs, limit controls, genre tags, and rating scores.
* **Movie Details & Related Content (`/movies/[movieId]`)**: Movie metadata display, active user star rating control ($0.5 \le rating \le 5.0$), and content-based recommendation showcase.
* **Recommendation Hub (`/recommendations`)**: Tabbed interface switching between Content-Based, Item-Based Collaborative, and Hybrid recommendation engines with real-time content/collaborative weight sliders ($w_{content} + w_{collab} = 1.0$).
* **User Management (`/users`)**: User registration form with client-side Zod validation, user lookup, inline detail edits, and deletion confirmation modal.
* **User Profile & Rating History (`/users/[userId]`)**: User profile dashboard displaying rated movies list, star rating modifier/deleter, and logged recommendation request history.

### 2. Tech Stack & Frontend Engineering
* **Framework**: Next.js 14 (App Router) with TypeScript strict mode.
* **Styling & UI**: Tailwind CSS (cinematic dark theme palette), Lucide React Icons.
* **Forms & Validation**: React Hook Form, Zod schema validation.
* **State Management**: `UserContext` for active user switching & persistent `localStorage` session, `ToastContext` for floating alerts.
* **Testing**: Vitest, React Testing Library, and jsdom (100% test pass rate).

---

## Quickstart Instructions

### 1. Setup Backend Environment (Python)

**macOS / Linux:**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r ml/requirements.txt

# Run database migrations
alembic upgrade head
```

### 2. Start FastAPI REST API Backend

```bash
uvicorn ml.api.main:app --reload --port 8000
```

- API Base URL: `http://localhost:8000`
- Swagger UI Documentation: `http://localhost:8000/docs`

### 3. Setup and Run Next.js Frontend Application

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start Next.js development server
npm run dev
```

- Web App Interface: `http://localhost:3000`

### 4. Running Verification Test Suites

```bash
# Run backend pytest suite (148 tests)
PYTHONPATH=. pytest ml/tests/

# Run frontend Vitest suite (8 tests)
cd frontend && npm test
```

### 5. Running Offline Evaluation Pipeline (Phase 6)

```bash
python3 ml/scripts/run_evaluation.py
```
