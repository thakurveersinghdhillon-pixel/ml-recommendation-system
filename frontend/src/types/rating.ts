export interface Rating {
  id: number;
  user_id: number;
  movie_id: number;
  rating: number;
  created_at: string;
  updated_at: string;
}

export interface RatingCreate {
  user_id: number;
  movie_id: number;
  rating: number;
}

export interface RatingListResponse {
  items: Rating[];
  total: number;
  limit: number;
  offset: number;
}
