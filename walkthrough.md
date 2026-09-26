# Hybrid ML Recommendation System — Comprehensive Project Walkthrough

This document provides a beginner-friendly, technical walkthrough of the **Hybrid ML Recommendation System**, covering Phase 1 (Data Environment & Validation), Phase 2 (Preprocessing & EDA), and Phase 3 (Content-Based Recommendation System).

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
Movie Catalog Metadata (title, genres)
          ↓
Prepare Text Content ("toy story adventure animation children comedy fantasy")
          ↓
TF-IDF Feature Matrix Construction (scikit-learn TfidfVectorizer)
          ↓
Pairwise Cosine Similarity Matrix (9,742 x 9,742)
          ↓
Title Lookup & Normalization (Case-insensitive & strip whitespace)
          ↓
Exclusion & Ranking (Exclude query movie, sort by similarity DESC, movieId ASC)
          ↓
Top-N Recommendation Output + Metadata-Grounded Explanations
```

### Feature Engineering & Text Representation
The content representation combines the cleaned title and genres into a single token string:
* Title `"Toy Story (1995)"` -> `"toy story"`
* Genres `"Adventure|Animation|Children|Comedy|Fantasy"` -> `"adventure animation children comedy fantasy"`
* Combined text: `"toy story adventure animation children comedy fantasy"`

If a movie has `"(no genres listed)"` or missing genres, it is handled as an empty string without crashing.

### TF-IDF & Cosine Similarity Mathematics

#### TF-IDF Vectorization
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
* $\text{TF}(t, d)$: Term frequency of word $t$ in movie text $d$.
* $\text{IDF}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$: Inverse document frequency, reducing the weight of common words and highlighting specific genre/title keywords.

#### Cosine Similarity
Pairwise similarity between two movie vectors $\vec{A}$ and $\vec{B}$:
$$\text{Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
* Score = `1.0`: Identical content feature direction (same movie or identical metadata).
* Score = `0.0`: Completely orthogonal / no shared terms or genres.

### Deterministic Title Lookup & Duplicate Handling
* Title strings are normalized by trimming whitespace and converting to lowercase:
  `"  TOY STORY (1995) "` -> `"toy story (1995)"`.
* If duplicate normalized titles exist in the dataset, the recommender deterministically maps to the **first occurrence** in catalog dataset order.

### Recommendation Sorting & Ranking Strategy
* Candidates are sorted using a deterministic two-key strategy:
  1. Primary key: `similarity_score` **DESCENDING**
  2. Secondary key (tie-breaker): `movieId` **ASCENDING**
* The query movie itself is strictly excluded from recommendation results.

### Explanation Support (`explain_recommendation`)
Explanations are generated deterministically from actual metadata evidence (no LLM dependence):
* Compares `genres` sets between source and recommended movies.
* Reports exact shared genres and the calculated cosine similarity score.

> **Example**:
> *"Recommended 'Toy Story 2 (1999)' because it shares 5 genre(s) ('Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy') with 'Toy Story (1995)'. Content similarity score: 0.7337."*

### Edge Case Handling Summary
| Edge Case | Handled Behavior |
| :--- | :--- |
| **Unknown Movie Title** | Raises `ValueError("Movie '...' not found in dataset catalog.")` |
| **Empty Movie Title String** | Raises `ValueError("Movie title must be a non-empty string.")` |
| **Invalid $N \le 0$** | Raises `ValueError("n must be a positive integer...")` |
| **$N > \text{Catalog Size}$** | Gracefully caps results to total available candidates without crashing |
| **Missing / Empty Genres** | Handled safely using empty genre sets, content text falls back to cleaned title |
| **Duplicate Titles** | First occurrence in dataset index order is selected deterministically |

### Trade-offs & Hybrid System Preparation
While content-based filtering excels at solving **item cold-start** (recommending new movies with 0 ratings) and providing **high interpretability**, it cannot capture user-user rating preferences or serendipitous discoveries. In future phases, this module will serve as a foundational component of the **Hybrid Recommendation System**.
