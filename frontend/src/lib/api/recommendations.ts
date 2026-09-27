import { fetchApi } from "./client";
import {
  ContentRecommendationRequest,
  CollaborativeRecommendationRequest,
  HybridRecommendationRequest,
  RecommendationResponse,
} from "@/types/recommendation";

export async function getContentRecommendations(
  req: ContentRecommendationRequest
): Promise<RecommendationResponse> {
  return fetchApi<RecommendationResponse>("/recommendations/content", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getCollaborativeRecommendations(
  req: CollaborativeRecommendationRequest
): Promise<RecommendationResponse> {
  return fetchApi<RecommendationResponse>("/recommendations/collaborative", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getHybridRecommendations(
  req: HybridRecommendationRequest
): Promise<RecommendationResponse> {
  return fetchApi<RecommendationResponse>("/recommendations/hybrid", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
