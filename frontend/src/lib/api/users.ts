import { fetchApi } from "./client";
import { User, UserCreate, UserUpdate } from "@/types/user";
import { RecommendationHistoryListResponse } from "@/types/recommendation";

export async function createUser(data: UserCreate): Promise<User> {
  return fetchApi<User>("/users", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getUser(userId: number): Promise<User> {
  return fetchApi<User>(`/users/${userId}`);
}

export async function updateUser(userId: number, data: UserUpdate): Promise<User> {
  return fetchApi<User>(`/users/${userId}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export async function deleteUser(userId: number): Promise<void> {
  return fetchApi<void>(`/users/${userId}`, {
    method: "DELETE",
  });
}

export async function getUserRecommendationHistory(
  userId: number,
  limit: number = 20,
  offset: number = 0
): Promise<RecommendationHistoryListResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });
  return fetchApi<RecommendationHistoryListResponse>(`/users/${userId}/recommendation-history?${params.toString()}`);
}
