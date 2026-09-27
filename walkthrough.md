# Hybrid ML Recommendation System — Comprehensive Project Walkthrough

This document provides a beginner-friendly, technical walkthrough of the **Hybrid ML Recommendation System**, covering Phase 1 (Data Environment & Validation), Phase 2 (Preprocessing & EDA), Phase 3 (Content-Based Recommendation System), and Phase 4 (Collaborative Filtering).

---

## 1. Phase 1 — Data Environment & Pipeline Validation

### Modular Project Layout
Phase 1 established a clean, modular Python environment separating configuration (`config.py`), data loading (`data_loader.py`), schema validation (`data_validator.py`), CLI scripts (`load_dataset.py`), and test suites (`test_data_loader.py`, `test_data_validator.py`).

### Idempotent Pipeline & Schema Validation
* Raw MovieLens files (`movies.csv`, `ratings.csv`) are downloaded once to `ml/data/raw/ml-latest-small/` and verified for column presence (`movieId`, `title`, `genres` and `userId`, `movieId`, `rating`, `timestamp`).
* Unnecessary re-downloading is skipped on subsequent executions.

---

## 2. Phase 2 — Data Preprocessing & Exploratory Data Analysis (EDA)

### Core Engineering Rules
1. **Raw Data Immutability**: Raw files in `ml/data/raw/` remain untouched. All processed outputs are written to `ml/data/processed/`.
2. **Release Year Extraction**: Regex `\((\d{4})\)` parses 4-digit release years from title strings (`Toy Story (1995)` -> `1995.0`), setting `NaN` if absent without dropping catalog movies.
3. **Genre Parsing & Explosion**: Pipe-separated genre strings (`Action|Adventure`) are parsed into lists. Multi-genre movies are exploded when computing per-genre metrics:
   - **Movie Count**: Unique catalog movies in that genre.
   - **Rating Count**: Total rating entries received by movies in that genre.

### Zero-Rating Preservation & Popularity Paradox
* Movie statistics perform a **`LEFT JOIN`** from catalog movies to rating aggregations. Unrated movies retain `rating_count = 0` and `average_rating = NaN` rather than being assigned fake 0.0 scores or dropped.
* Sorting strictly by average rating is misleading because a movie with 1 rating of 5.0 is not "better" than a movie with 300 ratings averaging 4.5. Popularity is sorted primarily by rating count.

### User-Item Interaction Matrix & Sparsity
* Matrix size: $610 \text{ users} \times 9,742 \text{ movies} = 5,942,620 \text{ possible interactions}$.
* Observed ratings: $100,836$.
* **Sparsity**: $1 - \frac{100,836}{5,942,620} = 98.30\%$.
* Unobserved interactions remain `NaN` (not `0`). High sparsity demonstrates the need for collaborative filtering (SVD/Matrix Factorization) and hybrid content models to solve cold-start issues.

---

## 3. Phase 3 — Content-Based Recommendation System

### Concept & Overview
Content-based recommendation generates recommendations by analyzing item attributes (metadata) rather than user rating behaviors. When a user queries a movie (e.g. *Toy Story (1995)*), the engine computes similarity scores against all other movies in the catalog using their textual metadata representation.

### Architecture & Pipeline Steps
```text
Movie Catalog Metadata (title, genres) → Text Content Representation → TF-IDF Vectorization → Cosine Similarity Matrix → Rank & Exclude Query Movie → Top-N Recommendations + Explanations
```

### Feature Engineering & Text Representation
The content representation combines the cleaned title and genres into a single token string (`"toy story adventure animation children comedy fantasy"`). Missing genres are handled as empty strings without crashing.

### TF-IDF & Cosine Similarity Mathematics
* **TF-IDF**: $\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$
* **Cosine Similarity**: $\text{Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$

---

## 4. Phase 4 — Collaborative Filtering

### Concept & Overview
Collaborative filtering generates recommendations by examining how users rate items across the dataset. Rather than relying on textual metadata (titles, genres), Item-Based Collaborative Filtering represents movies as rating vectors across all users and identifies movies with similar rating patterns.

```text
Movie A Rating Vector: [5, 4, 5, 0, 0]
Movie B Rating Vector: [4, 5, 4, 0, 0]
             ↓
High Cosine Similarity between User Rating Patterns
             ↓
Movies A and B are Collaboratively Similar
```

### Sparse User-Item Matrix Representation
The user interaction dataset is represented using a Scipy CSR sparse matrix `user_item_matrix` of shape $(610 \text{ users} \times 9,742 \text{ movies})$.
* Rows: `userId` (mapped 0 to $U-1$).
* Columns: `movieId` (mapped 0 to $M-1$).
* Values: User rating values ($0.5$ to $5.0$).
* Unobserved entries are preserved as sparse zeros without allocating memory.
* Duplicate `(userId, movieId)` pairs are aggregated deterministically using mean rating.

### Item-Item Cosine Similarity Calculation
The similarity between movie $i$ and movie $j$ is calculated as the cosine similarity between their user rating column vectors:
$$S_{i, j} = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\|_2 \|\vec{v}_j\|_2}$$
The item similarity matrix $S \in \mathbb{R}^{9,742 \times 9,742}$ is stored as dense `float32` requiring $\approx \mathbf{380 \text{ MB}}$ of memory.

### Normalized Mean-Centered Recommendation Scoring
To compute recommendations for user $u$, we evaluate candidate unrated movies $c$ using the normalized mean-centered formula:

$$\text{Score}(u, c) = \frac{\sum_{i \in \text{Rated}(u)} [S_{i, c} \times (r_{u, i} - \bar{r}_u)]}{\sum_{i \in \text{Rated}(u)} |S_{i, c}| + \epsilon}$$

1. **Mean-Centering ($r_{u, i} - \bar{r}_u$)**:
   - If user $u$ rated movie $i$ **above** their average rating $\bar{r}_u$ (e.g. rated 5.0 when mean is 3.5), it acts as a **positive signal** (+1.5), boosting candidates similar to $i$.
   - If user $u$ rated movie $i$ **below** their average rating (e.g. rated 1.0 when mean is 3.5), it acts as a **negative signal** (-2.5), penalizing candidates similar to $i$.
2. **Normalization ($\sum |S_{i, c}| + \epsilon$)**:
   - Normalizes by total similarity weight so candidates similar to many items are not artificially inflated. $\epsilon = 10^{-9}$ prevents division by zero.

### Mandatory Already-Rated Filtering
`recommend_for_user(user_id)` explicitly excludes all movies user $u$ has already rated in the training dataset (`user_rated_movies[user_id]`). This prevents recommending movies the user has already seen and rated.

### Deterministic Candidate Ranking
* Personalized Recommendations: `score` **DESCENDING**, tie-breaker `movieId` **ASCENDING**.
* Similar Movie Recommendations: `similarity_score` **DESCENDING**, tie-breaker `movieId` **ASCENDING**.

### Collaborative Explanation Support
`explain_user_recommendation(user_id, movie_id)` identifies the exact source movie $i$ rated by user $u$ that contributed most positively to candidate $c$'s score:

$$\text{Contribution} = S_{i, c} \times (r_{u, i} - \bar{r}_u)$$

> **Generated Explanation Sample**:
> *"Recommended 'Entertaining Angels: The Dorothy Day Story (1996)' (score: 0.6336) to User 1 because User 1 rated 'Schindler's List (1993)' 5.0 (above user average of 4.37). Item-item rating similarity between 'Schindler's List (1993)' and 'Entertaining Angels: The Dorothy Day Story (1996)' is 0.0777."*

### Cold-Start Limitations & Data Leakage Protection
* **Data Leakage Protection**: The model relies **only** on the DataFrame supplied to `fit()`, enabling clean train/test evaluation in Phase 6.
* **Cold-Start Limitations**:
  - **Unknown User**: A user without rating history cannot receive collaborative recommendations (`ValueError` is raised).
  - **Unknown Movie**: A movie without ratings cannot be evaluated (`ValueError` is raised).

---

## Phase 6 Walkthrough — Recommendation Model Evaluation

### Evaluation Setup & Methodology
- **Dataset**: MovieLens `ml-latest-small` (9,742 movies, 100,836 ratings).
- **Temporal Split**: 610 eligible users (100% of dataset). 80,419 training interactions, 20,417 test interactions. 0 excluded users.
- **Relevance Threshold**: Test ratings $\ge 4.0$. 18 test users had 0 relevant test items (excluded from macro-average recall denominators).
- **Candidate Pool Size**: Top 100 candidate items per model.

### Key Evaluation Metric Results

| Model | P@5 | R@5 | NDCG@5 | HR@5 | P@10 | R@10 | NDCG@10 | HR@10 | Coverage@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Content-Based** | **0.0105** | **0.0053** | **0.0126** | **0.0439** | **0.0072** | **0.0078** | **0.0111** | **0.0642** | **23.10%** |
| **Collaborative Filtering** | 0.0003 | 0.0000 | 0.0002 | 0.0017 | 0.0002 | 0.0000 | 0.0001 | 0.0017 | 9.47% |
| **Hybrid (50/50)** | 0.0023 | 0.0006 | 0.0027 | 0.0101 | 0.0021 | 0.0010 | 0.0025 | 0.0169 | 13.81% |

### Generated Artifacts
1. `ml/data/evaluation/metrics/model_comparison.csv`
2. `ml/data/evaluation/metrics/model_comparison.json`
3. `ml/data/evaluation/metrics/per_user_metrics.csv`
4. `ml/data/evaluation/plots/precision_at_k.png`
5. `ml/data/evaluation/plots/recall_at_k.png`
6. `ml/data/evaluation/plots/ndcg_at_k.png`
7. `ml/data/evaluation/plots/hit_rate_at_k.png`
8. `ml/data/evaluation/plots/catalog_coverage.png`

### Test Verification
All 122 unit tests pass cleanly:
```bash
python3 -m pytest ml/tests/
```

---

## Phase 7 Walkthrough — FastAPI Backend Development

### REST API Architecture & Lifespan Model Initialization
- **Lifespan Context Manager**: Dataset loading and fitting for `ContentBasedRecommender`, `ItemBasedCollaborativeRecommender`, and `HybridRecommender` occur **once** at application startup (`lifespan` in `ml/api/main.py`). Models are stored in `app.state`.
- **Dependency Injection**: Request handlers access ready models via `get_recommendation_service` dependency in `ml/api/dependencies.py`.
- **Pydantic Validation**: Strict request/response validation rejecting negative IDs, out-of-range `top_k` ($1 \le top\_k \le 50$), invalid weight sums, and invalid JSON payloads.

### Endpoint Smoke Verification Output
```text
Testing GET /api/v1/health...
  Status Code: 200
  Payload: {'status': 'healthy', 'service': 'movie-recommendation-api', 'models': {'content_based': 'ready', 'collaborative': 'ready', 'hybrid': 'ready'}}

Testing GET /api/v1/movies/search?q=toy...
  Status Code: 200
  Payload: {'query': 'toy', 'total_matches': 8, 'limit': 3, 'movies': [{'movie_id': 3086, 'title': 'Babes in Toyland (1934)'}, ...]}

Testing GET /api/v1/movies/1...
  Status Code: 200
  Payload: {'movie_id': 1, 'title': 'Toy Story (1995)', 'genres': ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy'], 'year': 1995}

Testing POST /api/v1/recommendations/content...
  Status Code: 200
  First Rec: Toy Story 2 (1999) Score: 0.7337

Testing POST /api/v1/recommendations/collaborative...
  Status Code: 200
  First Rec: Entertaining Angels: The Dorothy Day Story (1996) Score: 0.6336

Testing POST /api/v1/recommendations/hybrid...
  Status Code: 200
  First Rec: Entertaining Angels: The Dorothy Day Story (1996) Score: 0.5
```

---

## Phase 8 Walkthrough — PostgreSQL Database Integration

### Overview & Objectives
Phase 8 integrates a robust PostgreSQL database layer to persist `User` accounts, `Rating` entries, and `RecommendationHistory` logs for all API requests. The implementation utilizes **SQLAlchemy 2.x ORM**, **Alembic 1.x migrations**, **Pydantic v2 schemas**, and **FastAPI dependency injection**, while preserving 100% of the Phase 1–7 recommendation engines and offline evaluation pipelines intact.

### Database Architecture & Models
1. **`User` ORM Model (`users`)**:
   - `id`: Integer primary key (autoincrement).
   - `username`: String(50), unique, indexed, non-null.
   - `email`: String(100), unique, indexed, non-null.
   - `created_at` / `updated_at`: Timezone-aware UTC datetimes.
   - Relationships: `ratings` (cascade="all, delete-orphan"), `recommendation_history`.

2. **`Rating` ORM Model (`ratings`)**:
   - `id`: Integer primary key (autoincrement).
   - `user_id`: Foreign key referencing `users.id` (`ondelete="CASCADE"`).
   - `movie_id`: Integer, indexed.
   - `rating`: Float, validated $0.5 \le rating \le 5.0$ in 0.5 increments.
   - Constraints: `UniqueConstraint("user_id", "movie_id", name="uq_user_movie_rating")`.
   - Atomic `RatingRepository.upsert()` uses `SELECT FOR UPDATE` to avoid race conditions.

3. **`RecommendationHistory` ORM Model (`recommendation_history`)**:
   - `id`: Integer primary key.
   - `user_id`: Foreign key referencing `users.id` (`ondelete="SET NULL"`).
   - `model`: String(50) (`content_based`, `collaborative_filtering`, `hybrid`).
   - `seed_movie_id`, `top_k`, `content_weight`, `collaborative_weight`, `created_at`.
   - `recommendations`: JSON / JSONB list of recommended movie items.

### API Routes & Functionality Verified
- `POST /api/v1/users`: Create user (201 Created). Checks unique username/email, returning 409 Conflict if duplicate.
- `GET /api/v1/users/{id}`: Fetch user profile (200 OK or 404 Not Found).
- `PATCH /api/v1/users/{id}`: Partial update user profile (200 OK or 409 Conflict).
- `DELETE /api/v1/users/{id}`: Remove user and cascade delete ratings (204 No Content).
- `PUT /api/v1/ratings`: Atomic rating upsert (201 Created on insert, 200 OK on update).
- `GET /api/v1/users/{id}/ratings`: Paginated list of user ratings.
- `GET /api/v1/users/{id}/ratings/{movie_id}`: Fetch individual rating.
- `DELETE /api/v1/users/{id}/ratings/{movie_id}`: Delete individual rating (204 No Content).
- `GET /api/v1/users/{id}/recommendation-history`: Paginated list of logged recommendation requests.
- `GET /api/v1/health`: Extended to perform database readiness check (`SELECT 1`).

### Test Verification
All 148 automated tests pass across Phase 1 through Phase 8:
```bash
python3 -m pytest ml/tests/
```
```text
====================== 148 passed, 10 warnings in 15.27s =======================
```

---

## Phase 9 Walkthrough — Frontend Development and API Integration

### Overview & Objectives
Phase 9 delivers a complete, responsive Next.js web application for the ML Recommendation System, providing a interactive interface for movie discovery, recommendation generation across Content-Based, Collaborative Filtering, and Hybrid models, user profile management, and interactive star rating controls.

### Implementation Architecture
1. **Next.js App Router Structure (`frontend/src/app`)**:
   - `page.tsx`: Home dashboard featuring hero section, search quick-access, recommendation engine shortcuts, and API health status card.
   - `movies/page.tsx`: Catalog search with debounced text search, limit dropdown, genre pills, score badges, and pagination placeholders.
   - `movies/[movieId]/page.tsx`: Detailed view for catalog movies with interactive star rating control ($0.5 \le rating \le 5.0$) and a row of content-based similar movies.
   - `recommendations/page.tsx`: Recommendation generator supporting Content-Based, Item-Based Collaborative, and Hybrid recommendation engines with interactive content weight and collaborative weight sliders ($w_{\text{content}} + w_{\text{collab}} = 1.0$).
   - `users/page.tsx`: User administration page with registration form (Zod validation for username and email), user search lookup, inline detail modification, and modal confirmation for account deletion.
   - `users/[userId]/page.tsx`: Dedicated user profile dashboard listing user's rating history with inline score modification / deletion, and logged recommendation request history.

2. **State Management & Context Providers (`frontend/src/context`)**:
   - `UserContext.tsx`: Manages active user profile state with local persistence in `localStorage`, enabling Seamless switching between users across pages.
   - `ToastContext.tsx`: Provides non-intrusive floating toast notifications for user feedback on actions (rating added/updated, user registered/updated, error alerts).

3. **Type-Safe API Client (`frontend/src/lib/api`)**:
   - Modular REST client modules (`client.ts`, `movies.ts`, `recommendations.ts`, `users.ts`, `ratings.ts`, `health.ts`) wrapping backend endpoints with structured error handling via `CustomApiError`.

4. **Component Architecture (`frontend/src/components`)**:
   - `Navbar.tsx` & `Footer.tsx`: Cinematic dark header with active user selector badge, navigation links, and mobile drawer.
   - `StarRating.tsx`: Interactive half-star rating component supporting mouse hover previews and exact half-star increments ($0.5$ to $5.0$).
   - `MovieCard.tsx`: Standardized movie card with genre badges, release year, score pills, and quick rating action.
   - `SkeletonCard.tsx` & `ConfirmationModal.tsx`: Accessible loading placeholders and confirmation dialogs for destructive actions.

### Test & Build Verification

1. **Frontend Test Suite (Vitest + React Testing Library)**:
   ```bash
   cd frontend && npm test
   ```
   ```text
   RUN  v2.1.9 /Users/thakurveersingh/Documents/ML project/frontend

   ✓ src/test/api.test.ts (5)
   ✓ src/test/components.test.tsx (3)

   Test Files  2 passed (2)
        Tests  8 passed (8)
     Duration  813ms
   ```

2. **Production Build & ESLint Check**:
   ```bash
   cd frontend && npm run lint && npm run build
   ```
   ```text
   ✔ No ESLint warnings or errors
   ✓ Compiled successfully
   ✓ Prerendered 7 static/dynamic routes
   ```

3. **Backend Integration Verification**:
   ```bash
   PYTHONPATH=. .venv/bin/pytest ml/tests/ -v
   ```
   ```text
   ===================== 148 passed, 1 warning in 16.97s ======================
   ```
