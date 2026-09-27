import { fetchApi } from "./client";
import { Movie, MovieSearchResponse } from "@/types/movie";

export async function searchMovies(query: string, limit: number = 10): Promise<MovieSearchResponse> {
  const params = new URLSearchParams({
    q: query,
    limit: limit.toString(),
  });
  return fetchApi<MovieSearchResponse>(`/movies/search?${params.toString()}`);
}

export async function getMovieDetails(movieId: number): Promise<Movie> {
  return fetchApi<Movie>(`/movies/${movieId}`);
}
