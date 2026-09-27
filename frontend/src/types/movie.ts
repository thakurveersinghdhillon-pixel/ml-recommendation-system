export interface Movie {
  movie_id: number;
  title: string;
  genres: string[];
  year?: number | null;
}

export interface MovieSearchResponse {
  query: string;
  total_matches: number;
  limit: number;
  movies: Movie[];
}
