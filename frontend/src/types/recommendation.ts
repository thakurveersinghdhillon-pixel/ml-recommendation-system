export interface RecommendationItem {
  movie_id: number;
  title: string;
  genres: string[];
  score: number;
  content_score?: number | null;
  collaborative_score?: number | null;
  normalized_content_score?: number | null;
  normalized_collaborative_score?: number | null;
  explanation?: string | null;
}

export interface RecommendationResponse {
  model: string;
  user_id?: number | null;
  seed_movie_id?: number | null;
  seed_movie_title?: string | null;
  content_weight?: number | null;
  collaborative_weight?: number | null;
  recommendations: RecommendationItem[];
}

export interface ContentRecommendationRequest {
  movie_id: number;
  top_k?: number;
}

export interface CollaborativeRecommendationRequest {
  user_id: number;
  top_k?: number;
}

export interface HybridRecommendationRequest {
  user_id?: number | null;
  movie_id?: number | null;
  top_k?: number;
  content_weight?: number;
  collaborative_weight?: number;
}

export interface RecommendationHistoryItem {
  id: number;
  user_id?: number | null;
  model: string;
  seed_movie_id?: number | null;
  top_k: number;
  content_weight?: number | null;
  collaborative_weight?: number | null;
  created_at: string;
  recommendations: RecommendationItem[];
}

export interface RecommendationHistoryListResponse {
  items: RecommendationHistoryItem[];
  total: number;
  limit: number;
  offset: number;
}
