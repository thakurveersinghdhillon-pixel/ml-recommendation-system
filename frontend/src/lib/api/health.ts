import { fetchApi } from "./client";
import { HealthResponse } from "@/types/common";

export async function getHealth(): Promise<HealthResponse> {
  return fetchApi<HealthResponse>("/health");
}
