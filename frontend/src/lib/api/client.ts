import { CustomApiError } from "@/types/common";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_V1_PREFIX = "/api/v1";

interface FetchOptions extends RequestInit {
  timeoutMs?: number;
}

export async function fetchApi<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const { timeoutMs = 15000, headers, ...customConfig } = options;
  const url = `${API_BASE_URL}${API_V1_PREFIX}${endpoint}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const defaultHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };

  try {
    const response = await fetch(url, {
      ...customConfig,
      headers: {
        ...defaultHeaders,
        ...headers,
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.status === 204) {
      return {} as T;
    }

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const code = data?.error?.code || `HTTP_${response.status}`;
      const message = data?.error?.message || response.statusText || "An unexpected error occurred.";
      throw new CustomApiError(code, message, response.status);
    }

    return data as T;
  } catch (error: unknown) {
    clearTimeout(timeoutId);

    if (error instanceof CustomApiError) {
      throw error;
    }

    if (error instanceof Error && error.name === "AbortError") {
      throw new CustomApiError("REQUEST_TIMEOUT", "Network request timed out. Please try again.", 408);
    }

    const errorMsg = error instanceof Error ? error.message : "Failed to connect to backend service.";
    throw new CustomApiError("NETWORK_ERROR", errorMsg, 500);
  }
}
